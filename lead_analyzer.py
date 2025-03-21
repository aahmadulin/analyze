from openai import AsyncOpenAI
from typing import List, Dict
import json
import logging

logger = logging.getLogger(__name__)

class LeadAnalyzer:
    """Анализатор этапов диалога для квалификации лидов"""
    def __init__(self, openai_key: str):
        self.client = AsyncOpenAI(api_key=openai_key)
        
        # Промпты для каждого этапа
        self.stage_prompts = {
                "greeting": """Check the greeting stage:  
    1. Did the client agree to continue the conversation? (Yes/No)  
    2. Did the conversation continued?
    
    **Scoring rules:**  
    - If client agreed to continue conversation → "qualification" == True ()
    - If client didn't agree to continue conversation → "qualification" == True
    - If the conversation continued → "qualification" == True 
    
    **IMPORTANT**:
    It is strictly forbidden to write phrases into 'reason' that relate to the code!

    Answer in JSON (ответь на русском):  
    {
        "qualified": bool,
        "reason": "Ответ на русском языке напиши в конце анализа в пункт 'final_full_reason'"
    }""",

    "main_part": """Check the main part:  
    1. Were the details of the service discussed?  
    2. Did the client ask clarifying questions?  
    3. Did the client talk about their business?  

    **Scoring rules:**  
    - If at least one of the questions is answered "yes" → "qualification" == True  
    - If none of the questions are answered "yes" → "qualification" == False  

    **IMPORTANT**:
    It is strictly forbidden to write phrases into 'reason' that relate to the code!

    Answer in JSON (на русском):  
    {
        "qualified": bool,
        "reason": "Ответ на русском языке напиши в конце анализа в пункт 'final_full_reason'"
    }""",

    "closing": """Check the closing stage:  
    1. Is there an agreement on the next step?  
    2. Was there an exchange of contacts or an action plan?  
    3. Were future calls discussed?  

    **Scoring rules:**  
    - If at least one of the questions is answered "yes" → "qualification" == True  
    - If none of the questions are answered "yes" → "qualification" == False  

    **IMPORTANT**:
    It is strictly forbidden to write phrases into 'reason' that relate to the code!
    
    Answer in JSON (на русском):  
    {
        "qualified": bool,
        "reason": "Ответ на русском языке напиши в конце анализа в пункт 'final_full_reason'. Не отвечай жесткими фразами, типа 'лид неквалифицирован', 'лид нельзя назвать неквалифицированным'"

    In the end of the analysis, make a "kval_percentage" and write an int there which means how many percents do you think this lead is qualified
    }"""

    }


    async def extract_agreements(self, dialog_text: str) -> str:
        """Извлечение договоренностей через ChatGPT"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Ты эксперт по анализу деловых переговоров. Выяви конкретные договоренности:
- Сроки отправки КП
- Обязательства клиента по обратной связи
- Согласованные сроки выполнения работ
- Условия оплаты
- Другие взаимные обязательства

Ответь ТОЛЬКО в формате JSON:
{
    "agreement": "краткая сводка договоренностей"
}"""
                    },
                    {
                        "role": "user", 
                        "content": f"Диалог:\n{dialog_text}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("agreement", "Договоренности не выявлены")
            
        except Exception as e:
            logger.error(f"Ошибка извлечения договоренностей: {str(e)}")
            return "Не удалось определить договоренности"


    async def analyze_stage(self, stage_name: str, utterances: List[Dict]) -> Dict:
        """Анализ конкретного этапа диалога"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.stage_prompts[stage_name]},
                    {"role": "user", "content": "\n".join(
                        f"{utt['role']}: {utt['text']}" for utt in utterances
                    )}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            logger.error(f"Ошибка анализа этапа {stage_name}: {str(e)}")
            return {"qualified": False, "reason": "Ошибка анализа"}

    async def full_analysis(self, utterances: List[Dict]) -> Dict:
        """Полный анализ всех этапов"""
        analysis = {
            "stages": {},
            "score": 0,
            "final_verdict": "неквалифицированный",
            "final_reason": "",
            "final_full_reason": "",  # Сюда соберем все обоснования
            "kval_percentage": 0  # Процент квалификации, который определит ChatGPT
        }

        # Этап 1: Приветствие (первые 5 реплик)
        greeting_utts = [u for u in utterances if u.get("category") == "greeting"][:5]
        stage_result = await self.analyze_stage("greeting", greeting_utts)
        analysis["stages"]["greeting"] = stage_result
        if stage_result.get("qualified", False):
            analysis["score"] += 1

        # Этап 2: Основная часть
        main_utts = [u for u in utterances if u.get("category") == "main_part"]
        stage_result = await self.analyze_stage("main_part", main_utts)
        analysis["stages"]["main_part"] = stage_result
        if stage_result.get("qualified", False):
            analysis["score"] += 1

        # Этап 3: Заключение (последние 5 реплик)
        closing_utts = [u for u in utterances if u.get("category") == "closing"][-5:]
        stage_result = await self.analyze_stage("closing", closing_utts)
        analysis["stages"]["closing"] = stage_result
        if stage_result.get("qualified", False):
            analysis["score"] += 1

        # Финальный вердикт
        analysis["final_verdict"] = "квалифицированный" if analysis["score"] >= 2 else "неквалифицированный"
        analysis["final_reason"] = f"Пройдено этапов: {analysis['score']}/3"

        # Сбор всех обоснований в final_full_reason
        final_full_reason = []
        for stage, result in analysis["stages"].items():
            if "reason" in result:
                final_full_reason.append(result["reason"])
        analysis["final_full_reason"] = " ".join(final_full_reason)

        # Определение процента квалификации через ChatGPT
        try:
            # Формируем запрос для ChatGPT
            prompt = f"""Проанализируй диалог и определи процент квалификации лида (kval_percentage) на основе следующих данных:
            - Приветствие: {analysis["stages"]["greeting"].get("reason", "Нет данных")}
            - Основная часть: {analysis["stages"]["main_part"].get("reason", "Нет данных")}
            - Заключение: {analysis["stages"]["closing"].get("reason", "Нет данных")}

            Ответь только числом от 0 до 100, которое отражает процент квалификации лида.
            """
            
            # Отправляем запрос ChatGPT
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты эксперт по анализу диалогов. Определи процент квалификации лида, основываясь на 'final_full_reason'. Напиши только ответ в числовом виде. Ничего больше! Если в ключе 'final_verdict' стоит значение 'неквалифицированный' - тогда значение не может быть больше 50%."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            # Парсим ответ
            kval_percentage = int(response.choices[0].message.content.strip())
            analysis["kval_percentage"] = kval_percentage

        except Exception as e:
            logger.error(f"Ошибка определения процента квалификации: {str(e)}")
            analysis["kval_percentage"] = int((analysis["score"] / 3) * 100)  # Fallback, если ChatGPT не ответил

        return analysis