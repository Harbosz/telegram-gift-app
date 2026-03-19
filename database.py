import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

class GiftDatabase:
    def __init__(self, json_path='gifts.json', model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.json_path = json_path
        # Используем мультиязычную модель для поддержки русского/украинского
        self.model = SentenceTransformer(model_name)
        self.gifts = []
        self.embeddings = None
        self.load_data()

    def load_data(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.gifts = json.load(f)
        except FileNotFoundError:
            self.gifts = []
        
        if not self.gifts:
            print("База данных пуста.")
            return

        # Готовим тексты: название + описание
        texts = [f"{g['name']} {g['description']}" for g in self.gifts]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)
        print(f"Загружено {len(self.gifts)} подарков. Модель: мультиязычная.")

    def search_gifts(self, query, top_k=3):
        if not self.gifts:
            return []

        # Кодируем запрос пользователя
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Считаем сходство
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Берем индексы лучших совпадений
        top_results = np.argpartition(-cosine_scores.cpu(), range(min(top_k, len(self.gifts))))[:top_k]
        
        results = []
        for idx in top_results:
            gift = self.gifts[idx].copy()
            gift['score'] = float(cosine_scores[idx])
            results.append(gift)
            
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def add_gift(self, name, description, image_url):
        new_id = max([g['id'] for g in self.gifts]) + 1 if self.gifts else 1
        new_gift = {
            "id": new_id,
            "name": name,
            "description": description,
            "image_url": image_url
        }
        self.gifts.append(new_gift)
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.gifts, f, indent=2, ensure_ascii=False)
        
        self.load_data()
        return new_gift

class ShowcaseDatabase:
    def __init__(self, json_path='showcases.json'):
        self.json_path = json_path
        self.showcases = []
        self.load_data()

    def load_data(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.showcases = json.load(f)
        except FileNotFoundError:
            self.showcases = []

    def save_showcase(self, user_id, name, slots):
        import time
        new_showcase = {
            "id": int(time.time()),
            "user_id": user_id,
            "name": name or f"Collection #{len(self.showcases) + 1}",
            "slots": slots,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.showcases.append(new_showcase)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.showcases, f, indent=2, ensure_ascii=False)
        return new_showcase

    def get_user_showcases(self, user_id):
        return [s for s in self.showcases if s['user_id'] == user_id]
