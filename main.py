import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from pydantic import BaseModel
from openai import AsyncOpenAI
from datetime import timezone
import time 
import random
import tempfile
from lead_analyzer import LeadAnalyzer
from dialog_stage_classifier import DialogStageClassifier

# Инициализация приложения FastAPI
app = FastAPI(title="Анализатор диалогов с ChatGPT")

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger(__name__)


# Модель для конфигурации
class AppConfig(BaseModel):
    assemblyai_key: str
    openai_key: str

# class AnalysisRequest(BaseModel):
#     id: str
#     manager_name: str

class RoleClassifier:
    """Классификатор ролей с использованием ChatGPT"""
    def __init__(self, openai_key: str):
        self.client = AsyncOpenAI(api_key=openai_key)
        self.system_prompt = """Ты эксперт по анализу телефонных переговоров. 
Определи роли:
- Менеджер (сотрудник компании, который предлагает услуги/товары)
- Клиент (потенциальный покупатель или заинтересованное лицо)

Правила определения:
1. В начале диалога может проигрываться песня. Это рингтон ожидания ответа пользователя. У менеджеров такого нет. Если ты видишь, что первая идет песня - тогда role == Клиент.
1. Менеджер обычно называет свое имя и должность
2. Менеджер задает уточняющие вопросы о бизнесе клиента. В то время как клиент обычно рассказывает о своем бизнесе и ситуации в его продажах.
3. Клиент запрашивает информацию о продукте
4. Менеджер предлагает коммерческое предложение
5. Клиент может рассказывать о своем бизнесе, о внутренних процессах в его компании

Если в тексте JSON, который ты анализируешь изначально передается один speaker, то выяви роль по контексту, кто клиент, а кто менеджер

Ответь ТОЛЬКО в JSON формате:
{
    "roles": {
        "A": "Менеджер/Клиент",
        "B": "Менеджер/Клиент"
    }
}

Перепроверь свои ответы по контексту и смыслу и выдай правильный ответ в поле "roles"

"""


    async def determine_roles(self, utterances: List[Dict]) -> Dict:
        """Определение ролей через ChatGPT"""
        try:
            # Формируем диалог с явным указанием спикеров
            dialog = "\n".join([f"Speaker {utt['speaker']}: {utt['text']}" for utt in utterances])
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Диалог:\n{dialog}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)
            logger.debug(f"Ответ от GPT: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            valid_roles = {"Менеджер", "Клиент"}
            for speaker, role in result["roles"].items():
                if role not in valid_roles:
                    raise ValueError(f"Недопустимая роль '{role}' для спикера {speaker}")
            
            return {k.upper(): v for k, v in result.get("roles", {}).items()}
            
        except Exception as e:
            logger.error(f"Ошибка определения ролей: {str(e)}")
            return {}


    async def split_monologue_into_dialog(self, monologue: str) -> List[Dict]:
        """Разделение монолога на диалог с помощью ChatGPT"""
        try:
            split_prompt = """Ты эксперт по анализу текстов. Раздели следующий монолог на диалог между двумя спикерами (Менеджер и Клиент). 
            Ответь в формате JSON, где каждая реплика будет иметь поля:
            - "speaker": "A" или "B" (A - Менеджер, B - Клиент)
            - "text": текст реплики

            Пример:
            {
                "utterances": [
                    {"speaker": "A", "text": "Здравствуйте, меня зовут Иван, я менеджер компании."},
                    {"speaker": "B", "text": "Здравствуйте, я хочу узнать о ваших услугах."}
                ]
            }
            """

            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": split_prompt},
                    {"role": "user", "content": monologue}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            logger.debug(f"Ответ от GPT (разделение монолога): {json.dumps(result, indent=2, ensure_ascii=False)}")

            return result.get("utterances", [])
            
        except Exception as e:
            logger.error(f"Ошибка разделения монолога: {str(e)}")
            return []


async def upload_file(file: UploadFile, api_key: str) -> str:
    """Загрузка файла на AssemblyAI"""
    try:
        logger.info(f"Начало загрузки файла {file.filename}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.assemblyai.com/v2/upload",
                headers={'authorization': api_key},
                data=file.file
            ) as response:
                response.raise_for_status()
                result = await response.json()
                logger.debug(f"Ответ от AssemblyAI (upload): {result}")
                return result['upload_url']
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети при загрузке: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка соединения с сервисом")
    except Exception as e:
        logger.error(f"Общая ошибка загрузки: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка загрузки файла")


async def transcribe_audio(audio_url: str, api_key: str) -> Dict:
    try:
        logger.info(f"Начало транскрипции для {audio_url}")
        async with aiohttp.ClientSession() as session:
            transcript_data = {
                "audio_url": audio_url,
                "speaker_labels": True,
                "language_code": "ru",
                "disfluencies": False
            }
            
            response = await session.post(
                "https://api.assemblyai.com/v2/transcript",
                json=transcript_data,
                headers={'authorization': api_key}
            )
            response.raise_for_status()
            transcript = await response.json()
            transcript_id = transcript['id']
            logger.debug(f"Создана транскрипция ID: {transcript_id}")

            retries = 60  
            while retries > 0:
                await asyncio.sleep(2)
                status_response = await session.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers={'authorization': api_key}
                )
                status_data = await status_response.json()
                logger.debug(f"Статус транскрипции: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    return status_data
                if status_data['status'] == 'failed':
                    logger.error(f"Ошибка транскрипции: {status_data.get('error', 'Unknown error')}")
                    raise ValueError(status_data.get('error', 'Транскрипция не удалась'))
                
                retries -= 1

            raise TimeoutError("Таймаут транскрипции: превышено время ожидания")
            
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети при транскрипции: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка соединения с сервисом")
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(e)}")


def process_transcript(transcript_data: Dict) -> List[Dict]:
    """Обработка результатов транскрипции"""
    return [
        {
            "speaker": utt["speaker"].strip().upper(),
            "text": utt["text"],
            "start": utt["start"],
            "end": utt["end"]
        }
        for utt in transcript_data.get("utterances", [])
    ]


def load_config() -> Optional[AppConfig]:
    try:
        config_path = Path("config.json")
        if not config_path.exists():
            logger.error("Файл config.json не найден")
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            required = {"assemblyai_key", "openai_key"}
            if missing := required - set(config_data.keys()):
                logger.error(f"Отсутствуют поля: {missing}")
                return None
                
            return AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфига: {str(e)}")
        return None


async def send_transcription_to_server(
    transcription_data: Dict,
    audio_file: UploadFile,
    id_value: str,
    manager_name: str
) -> Dict:
    """Отправляет данные на внешний сервер"""
    try:
        endpoint_url = f"http://185.207.0.3:5000/transcribe/{id_value}"
        transcription_data["manager"] = manager_name

        # Создаем FormData и добавляем поля
        form_data = aiohttp.FormData()
        
        # Добавляем JSON-данные
        form_data.add_field(
            name="data",
            value=json.dumps(transcription_data),
            content_type="application/json"
        )
        
        # Добавляем аудиофайл напрямую из UploadFile
        form_data.add_field(
            name="audio_file",
            value=await audio_file.read(),
            filename=audio_file.filename,
            content_type=audio_file.content_type
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, data=form_data) as response:
                response.raise_for_status()
                return await response.json()
                
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сети: {str(e)}")
    except Exception as e:
        logger.error(f"Ошибка отправки: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка отправки: {str(e)}")


async def analyze_parasite_words(manager_text: str, openai_key: str) -> str:
    """
    Анализирует текст менеджера на наличие слов-паразитов.
    
    :param manager_text: Текст реплик менеджера.
    :param openai_key: API-ключ OpenAI.
    :return: Результат анализа в виде строки.
    """
    try:
        client = AsyncOpenAI(api_key=openai_key)

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты эксперт по анализу речи. Проанализируй текст диалога и определи, какие слова-паразиты использует менеджер. Если слов паразитов не выявлено, то нужно написать о том, что менеджер молодец и не употреблял слова-паразитов."},
                {"role": "user", "content": f"Вот текст диалога менеджера:\n{manager_text}"}
            ],
            temperature=0.0,
            max_tokens=500
        )

        analysis_result = response.choices[0].message.content
        return analysis_result
    
    except Exception as e:
        logger.error(f"Ошибка анализа слов-паразитов: {str(e)}")
        return "Не удалось проанализировать слова-паразиты."


@app.post("/analyze", summary="Анализ аудио диалога")
async def analyze_dialog(
    id: str = Query(..., description="Уникальный идентификатор"),
    manager_name: str = Body(..., embed=True),
    file: UploadFile = File(...)
):
    """Анализ аудиозаписи с определением ролей и оценкой лида"""
    
    call_start_time = datetime.now(timezone.utc)
    
    try:
        config = load_config()
        if not config:
            raise HTTPException(status_code=500, detail="Ошибка конфигурации")

        # 1. Загрузка и транскрибация аудио
        audio_url = await upload_file(file, config.assemblyai_key)
        transcript = await transcribe_audio(audio_url, config.assemblyai_key)
        utterances = process_transcript(transcript)
        logger.debug(f"Обработанные реплики: {json.dumps(utterances, ensure_ascii=False)}")

        # 2. Обработка монолога при необходимости
        if len({utt["speaker"] for utt in utterances}) == 1:
            monologue = " ".join(utt["text"] for utt in utterances)
            role_classifier = RoleClassifier(config.openai_key)
            dialog_utterances = await role_classifier.split_monologue_into_dialog(monologue)
            utterances = [{"speaker": utt["speaker"], "text": utt["text"], "start": 0, "end": 0} 
                        for utt in dialog_utterances]

        # 3. Определение ролей
        role_classifier = RoleClassifier(config.openai_key)
        roles = await role_classifier.determine_roles(utterances)
        if not roles:
            raise HTTPException(status_code=500, detail="Не удалось определить роли")

        # 4. Формирование подготовленных данных
        prepared_utterances = []
        analyzed_data = []
        for utt in utterances:
            role = roles.get(utt["speaker"], "Неизвестно")
            analyzed_data.append({
                "text": utt["text"],
                "speaker": utt["speaker"],
                "role": role,
                "start_time": utt["start"],
                "end_time": utt["end"]
            })
            prepared_utterances.append({
                "role": role,
                "text": utt["text"],
                "speaker": utt["speaker"]
            })

        # 5. Классификация этапов диалога
        try:
            stage_classifier = DialogStageClassifier(
                api_key=config.openai_key,
                max_concurrent_requests=3
            )
            prepared_utterances = await stage_classifier.classify_utterances(prepared_utterances)
        except Exception as e:
            logger.error(f"Ошибка классификации этапов: {str(e)}", exc_info=True)
            prepared_utterances = [utt | {"category": "unknown"} for utt in prepared_utterances]

        # 6. Анализ лида и договоренностей
        lead_analyzer = LeadAnalyzer(config.openai_key)
        try:
            lead_analysis = await lead_analyzer.full_analysis(prepared_utterances)
            dialog_text = "\n".join(f"{utt['role']}: {utt['text']}" for utt in prepared_utterances)
            agreement_analysis = await lead_analyzer.extract_agreements(dialog_text)
        except Exception as e:
            logger.error(f"Ошибка анализа: {str(e)}")
            lead_analysis = {"error": str(e)}
            agreement_analysis = "Ошибка анализа"

        # 7. Анализ слов-паразитов
        manager_text = "\n".join(
            utt["text"] for utt in prepared_utterances 
            if utt.get("role") == "Менеджер"
        )
        parasite_words_analysis = await analyze_parasite_words(manager_text, config.openai_key)

        # 8. Формирование результата
        result = {
            "id": id,
            "manager": manager_name,
            "status": "success",
            "transcript": transcript["text"],
            "role_analysis": analyzed_data,
            "lead_analysis": lead_analysis,
            "agreement": agreement_analysis,
            "parasite_words_analysis": parasite_words_analysis,
            "call_start_time": call_start_time.isoformat(), 
            "call_end_time": datetime.now(timezone.utc).isoformat()
        }
        
        # 9. Посылка на эндпоинт
        await file.seek(0)
        await send_transcription_to_server(result, file, id, manager_name)
        
        return result
    
    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
