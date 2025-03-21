from openai import AsyncOpenAI
from typing import List, Dict
import logging
from asyncio import Semaphore

logger = logging.getLogger(__name__)

class DialogStageClassifier:
    def __init__(self, api_key: str, max_concurrent_requests: int = 5):
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = Semaphore(max_concurrent_requests)
    
    async def _classify_single_utterance(self, text: str, role: str) -> str:
        categories = {
            "greeting": "Приветствие/Введение",
            "main_part": "Основная часть/Продажа",
            "closing": "Заключение/Фиксация договоренностей"
        }
        
        system_prompt = f"""Определи категорию реплики. Варианты:
        1. {categories['greeting']} - установление контакта, приветствие
        2. {categories['main_part']} - обсуждение деталей, предложение услуг
        3. {categories['closing']} - завершение разговора, договоренности
        
        Ответ только номером категории (1, 2 или 3) без пояснений."""

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Реплика ({role}): {text}"}
                    ],
                    temperature=0.0,  # Уменьшил температуру для стабильности
                    max_tokens=2
                )
                
                raw_response = response.choices[0].message.content.strip()
                # Фильтрация недопустимых ответов
                if raw_response not in {"1", "2", "3"}:
                    logger.warning(f"Некорректный ответ GPT: '{raw_response}' для текста: '{text}'")
                    return "unknown"
                return raw_response
                
            except Exception as e:
                logger.error(f"Ошибка классификации: {str(e)}", exc_info=True)
                return "unknown"

    def _map_category(self, response: str) -> str:
        mapping = {
            "1": "greeting",
            "2": "main_part",
            "3": "closing"
        }
        return mapping.get(response.strip(), "unknown")

    async def classify_utterances(self, utterances: List[Dict]) -> List[Dict]:
        processed = []
        for idx, utterance in enumerate(utterances):
            try:
                processed_utt = dict(utterance)
                text = processed_utt.get("text", "")
                role = processed_utt.get("role", "unknown")
                
                if not text.strip():
                    processed_utt["category"] = "empty"
                    continue
                
                raw_response = await self._classify_single_utterance(text, role)
                processed_utt["category"] = self._map_category(raw_response)
                logger.debug(f"Результат классификации [{idx}]: {processed_utt['category']}")
                
            except Exception as e:
                logger.error(f"Ошибка обработки реплики: {str(e)}", exc_info=True)
                processed_utt["category"] = "unknown"
            
            processed.append(processed_utt)
        
        return processed