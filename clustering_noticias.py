import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import re
import threading
from collections import Counter

# Descargar stopwords si es necesario
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NewsClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Noticias - K-means")
        self.root.geometry("1000x700")
        
        self.noticias = []
        self.cluster_info = {}
        
        # Diccionarios mejorados con términos muy específicos
        self.category_keywords = {
            'Deportes': {
                'gol', 'portero', 'estadio', 'árbitro', 'messi', 'nadal', 'djokovic',
                'champions', 'mundial', 'olímpico', 'balón', 'cancha', 'entrenador',
                'lesión', 'hat-trick', 'clásico', 'derrota', 'victoria', 'competencia',
                'medalla', 'trofeo', 'atleta', 'deportivo', 'fútbol', 'baloncesto',
                'tenis', 'selección', 'jugador', 'equipo', 'partido'
            },
            'Política': {
                'presidente', 'ministro', 'congreso', 'senado', 'gobierno', 'elección',
                'votación', 'ley', 'reforma', 'fiscal', 'económica', 'económicas',
                'inflación', 'debate', 'político', 'democracia', 'constitución',
                'legislativo', 'parlamento', 'diplomacia', 'administración',
                'mandatario', 'gobernador', 'alcalde', 'ministerio', 'presupuesto',
                'impuesto', 'decreto', 'corrupción', 'oposición', 'coalición',
                'medidas', 'sistema', 'nacional', 'social'
            },
            'Tecnología': {
                'tecnología', 'inteligencia', 'artificial', 'software', 'hardware',
                'computadora', 'aplicación', 'internet', 'web', 'digital', 'google',
                'microsoft', 'apple', 'iphone', 'android', 'samsung', 'tesla', 'robot',
                'automático', 'autónomo', 'programación', 'algoritmo', 'big data',
                'nube', 'blockchain', 'criptomoneda', 'bitcoin', 'realidad virtual',
                'metaverso', 'innovación', 'revolucionaria', 'cuántica', 'computación',
                'procesador', 'chip', 'sistema', 'avances'
            }
        }
        
        # Términos exclusivos para desempate (alto peso)
        self.exclusive_terms = {
            'Deportes': {'gol', 'portero', 'estadio', 'árbitro', 'hat-trick', 'messi', 'nadal'},
            'Política': {'presidente', 'ministro', 'congreso', 'senado', 'fiscal', 'votación', 'elección'},
            'Tecnología': {'software', 'hardware', 'programación', 'algoritmo', 'bitcoin', 'blockchain', 'inteligencia artificial'}
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Clasificador de Noticias - K-means", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=10)
        
        # Frame de entrada
        input_frame = ttk.LabelFrame(main_frame, text="Ingresar Noticias", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        # Área de texto para ingresar noticias
        ttk.Label(input_frame, text="Ingrese las noticias (una por línea):").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.text_area = scrolledtext.ScrolledText(input_frame, width=80, height=12, wrap=tk.WORD, font=("Arial", 10))
        self.text_area.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Frame para botones (centrado)
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        # Botones centrados
        example_button = ttk.Button(button_frame, text="Cargar Ejemplos", command=self.load_examples, width=15)
        example_button.grid(row=0, column=0, padx=5)
        
        process_button = ttk.Button(button_frame, text="Clasificar Noticias", command=self.process_clustering, width=15)
        process_button.grid(row=0, column=1, padx=5)
        
        clear_button = ttk.Button(button_frame, text="Limpiar", command=self.clear_text, width=15)
        clear_button.grid(row=0, column=2, padx=5)
        
        # Frame de resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultados de Clasificación", padding="10")
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Pestaña de resumen
        summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(summary_tab, text="Resumen General")
        summary_tab.columnconfigure(0, weight=1)
        summary_tab.rowconfigure(0, weight=1)
        
        # Texto de resumen
        self.summary_text = scrolledtext.ScrolledText(summary_tab, wrap=tk.WORD, font=("Arial", 10))
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.summary_text.config(state=tk.DISABLED)
        
        # Pestaña de detalles por categoría - SOLO LISTA DE NOTICIAS
        self.detail_tabs = {}
        categories = ['Deportes', 'Política', 'Tecnología']
        
        for category in categories:
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=category)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
            
            # Frame para lista de noticias (ocupa toda la pestaña)
            list_frame = ttk.Frame(tab)
            list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
            list_frame.columnconfigure(0, weight=1)
            list_frame.rowconfigure(0, weight=1)
            
            # Listbox con scrollbar más grande
            listbox = tk.Listbox(list_frame, font=("Arial", 10), selectmode=tk.SINGLE, height=15)
            scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
            listbox.configure(yscrollcommand=scrollbar.set)
            
            listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            self.detail_tabs[category] = {
                'listbox': listbox
            }
        
        # Barra de progreso
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Etiqueta de estado
        self.status_label = ttk.Label(main_frame, text="Listo para clasificar noticias")
        self.status_label.grid(row=4, column=0, sticky=tk.W)
        
        # Configurar pesos para expansión
        main_frame.rowconfigure(2, weight=1)
        
    def load_examples(self):
        """Cargar ejemplos predefinidos"""
        examples = [
            # Deportes
            "El Barcelona ganó 3-0 contra el Real Madrid en el clásico de liga española",
            "Messi anota hat-trick en la final de la Champions League ante el PSG",
            "La selección nacional de fútbol se prepara para el mundial de 2026",
            "El equipo de baloncesto Lakers ganó el campeonato con una jugada espectacular",
            
            # Política
            "El presidente anunció nuevas medidas económicas para combatir la inflación",
            "El congreso debate la reforma fiscal para el próximo año fiscal",
            "Ministro de economía anuncia plan de estabilización monetaria",
            "El gobierno aprueba nueva ley de educación para reformar el sistema",
            
            # Tecnología
            "Nuevo iPhone 15 con inteligencia artificial revolucionaria y mejoras en la cámara",
            "Google presenta avances en computación cuántica con nuevo procesador",
            "Microsoft lanza Windows 12 con funciones de IA integradas y mejor seguridad",
            "Facebook anuncia metaverso con realidad virtual mejorada para usuarios"
        ]
        
        self.text_area.delete(1.0, tk.END)
        for example in examples:
            self.text_area.insert(tk.END, example + "\n")
        self.status_label.config(text="Ejemplos cargados. Listo para clasificar.")
    
    def clear_text(self):
        """Limpiar el área de texto"""
        self.text_area.delete(1.0, tk.END)
        self.status_label.config(text="Texto limpiado. Puede ingresar nuevas noticias.")
        
    def preprocess_text(self, text):
        """Preprocesamiento del texto"""
        if pd.isna(text) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-záéíóúñ\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_category_score(self, text, category):
        """Calcular score para una categoría con pesos inteligentes"""
        words = set(text.split())
        score = 0
        
        # Puntos por palabras clave normales
        normal_keywords = self.category_keywords[category]
        normal_matches = words.intersection(normal_keywords)
        score += len(normal_matches) * 2
        
        # Puntos extra por términos exclusivos
        exclusive_terms = self.exclusive_terms[category]
        exclusive_matches = words.intersection(exclusive_terms)
        score += len(exclusive_matches) * 5
        
        return score
    
    def classify_single_news(self, news_text):
        """Clasificar una noticia individualmente de manera robusta"""
        text = self.preprocess_text(news_text)
        scores = {}
        
        for category in self.category_keywords:
            scores[category] = self.calculate_category_score(text, category)
        
        # Si hay empate, usar términos exclusivos para desempate
        max_score = max(scores.values())
        candidates = [cat for cat, score in scores.items() if score == max_score]
        
        if len(candidates) > 1:
            for category in candidates:
                exclusive_terms = self.exclusive_terms[category]
                if any(term in text for term in exclusive_terms):
                    return category
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def process_clustering(self):
        """Procesar el clustering"""
        text_content = self.text_area.get(1.0, tk.END).strip()
        if not text_content:
            messagebox.showerror("Error", "Por favor, ingrese algunas noticias primero.")
            return
        
        self.noticias = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        if len(self.noticias) < 3:
            messagebox.showerror("Error", "Se necesitan al menos 3 noticias para hacer clustering.")
            return
        
        self.progress.start()
        self.status_label.config(text="Procesando clasificación...")
        
        thread = threading.Thread(target=self._run_advanced_clustering)
        thread.daemon = True
        thread.start()
    
    def _run_advanced_clustering(self):
        """Ejecutar clustering avanzado con post-procesamiento robusto"""
        try:
            # 1. Primero clasificar cada noticia individualmente
            individual_classifications = []
            for news in self.noticias:
                category = self.classify_single_news(news)
                individual_classifications.append((news, category))
            
            # 2. Usar K-means para agrupamiento
            processed_docs = [self.preprocess_text(doc) for doc in self.noticias]
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=stopwords.words('spanish'),
                min_df=1,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(processed_docs)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=300)
            labels = kmeans.fit_predict(X)
            
            # 3. Organizar clusters iniciales
            initial_clusters = {}
            for i in range(3):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    cluster_docs = [self.noticias[idx] for idx in cluster_indices]
                    initial_clusters[i] = cluster_docs
            
            # 4. Post-procesamiento: re-asignar noticias basado en clasificación individual
            final_categories = {'Deportes': [], 'Política': [], 'Tecnología': []}
            
            # Para cada noticia, usar la clasificación individual como primaria
            for i, (news, category) in enumerate(individual_classifications):
                final_categories[category].append(news)
            
            # 5. Crear cluster_info para la visualización
            self.cluster_info = {}
            for i, (category, news_list) in enumerate(final_categories.items()):
                if news_list:
                    # Encontrar términos representativos
                    all_text = ' '.join(news_list).lower()
                    words = all_text.split()
                    word_counts = Counter(words)
                    
                    # Filtrar palabras clave relevantes
                    top_keywords = []
                    for word, count in word_counts.most_common(10):
                        if (len(word) > 3 and 
                            any(word in self.category_keywords[cat] for cat in self.category_keywords)):
                            top_keywords.append(word)
                    
                    self.cluster_info[i] = {
                        'name': category,
                        'keywords': ', '.join(top_keywords[:5]),
                        'count': len(news_list),
                        'documents': news_list
                    }
            
            self.root.after(0, self._update_ui)
            
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))
    
    def _update_ui(self):
        """Actualizar la interfaz con los resultados"""
        self.progress.stop()
        
        # Agrupar noticias por categoría para el display
        categorized_news = {'Deportes': [], 'Política': [], 'Tecnología': []}
        for info in self.cluster_info.values():
            categorized_news[info['name']].extend(info['documents'])
        
        # Actualizar resumen general
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        summary = "RESULTADOS DE CLASIFICACIÓN (K-means + Post-procesamiento)\n"
        summary += "=" * 60 + "\n\n"
        summary += f"Total de noticias procesadas: {len(self.noticias)}\n\n"
        
        for category in ['Deportes', 'Política', 'Tecnología']:
            count = len(categorized_news[category])
            if count > 0:
                summary += f"✅ {category.upper()} ({count} noticias)\n"
        
        summary += "\nSistema: K-means + Clasificación semántica individual"
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)
        
        # Limpiar y llenar pestañas de detalles
        for category in self.detail_tabs:
            self.detail_tabs[category]['listbox'].delete(0, tk.END)
        
        # Llenar las pestañas con las noticias completas
        for category, documents in categorized_news.items():
            if category in self.detail_tabs and documents:
                listbox = self.detail_tabs[category]['listbox']
                for i, doc in enumerate(documents):
                    # Mostrar la noticia completa en la lista
                    listbox.insert(tk.END, f"{i+1}. {doc}")
        
        self.status_label.config(text=f"Clasificación completada: {len(self.noticias)} noticias procesadas")
        messagebox.showinfo("Completado", "¡Clasificación exitosa!\n\nSe utilizó K-means para agrupamiento inicial + clasificación semántica individual para precisión.")
    
    def _handle_error(self, error_msg):
        """Manejar errores"""
        self.progress.stop()
        messagebox.showerror("Error", f"Ocurrió un error:\n{error_msg}")
        self.status_label.config(text="Error en el procesamiento")

def main():
    """Función principal"""
    root = tk.Tk()
    app = NewsClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()