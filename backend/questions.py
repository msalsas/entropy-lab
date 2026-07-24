"""Candidate question pool for the entropy divergence experiment.

The pool mixes Spanish and English questions across several categories
(factual, creative, ambiguous, open). Divergence between models is a
property of (question x model A x model B), so the pool deliberately
includes questions where a multilingual model and an English-centric
model are expected to differ in next-token uncertainty.
"""

QUESTION_POOL = [
    # --- Spanish, factual ---
    {"id": "es_fact_capital", "lang": "es", "category": "factual",
     "text": "¿Cuál es la capital de Portugal?"},
    {"id": "es_fact_river", "lang": "es", "category": "factual",
     "text": "¿Qué río pasa por la ciudad de Sevilla?"},
    {"id": "es_fact_author", "lang": "es", "category": "factual",
     "text": "¿Quién escribió la novela 'Cien años de soledad'?"},
    {"id": "es_fact_bodies", "lang": "es", "category": "factual",
     "text": "¿Cuántos huesos tiene aproximadamente el cuerpo humano adulto?"},
    # --- Spanish, open / creative ---
    {"id": "es_crea_story", "lang": "es", "category": "creative",
     "text": "Escribe el comienzo de un cuento sobre un farero que recibe una carta misteriosa."},
    {"id": "es_crea_poem", "lang": "es", "category": "creative",
     "text": "Improvisa cuatro versos sobre la lluvia en una ciudad vacía."},
    {"id": "es_open_advice", "lang": "es", "category": "open",
     "text": "¿Qué consejo le darías a alguien que empieza a aprender un idioma nuevo?"},
    {"id": "es_open_future", "lang": "es", "category": "open",
     "text": "¿Cómo imaginas el transporte urbano dentro de cincuenta años?"},
    # --- Spanish, ambiguous ---
    {"id": "es_amb_banco", "lang": "es", "category": "ambiguous",
     "text": "Me senté en el banco. ¿Dónde crees que estoy?"},
    {"id": "es_amb_capa", "lang": "es", "category": "ambiguous",
     "text": "Explica qué es una capa, dando al menos dos significados distintos."},
    {"id": "es_amb_moral", "lang": "es", "category": "ambiguous",
     "text": "¿Está bien mentir para no herir a alguien?"},
    {"id": "es_amb_math", "lang": "es", "category": "ambiguous",
     "text": "Si un tren sale a las ocho y otro a las nueve, ¿cuál llega antes?"},
    # --- English, factual ---
    {"id": "en_fact_planet", "lang": "en", "category": "factual",
     "text": "Which planet has the most moons?"},
    {"id": "en_fact_element", "lang": "en", "category": "factual",
     "text": "What chemical element has the symbol 'W'?"},
    {"id": "en_fact_painter", "lang": "en", "category": "factual",
     "text": "Who painted the ceiling of the Sistine Chapel?"},
    {"id": "en_fact_ocean", "lang": "en", "category": "factual",
     "text": "What is the deepest ocean on Earth?"},
    # --- English, open / creative ---
    {"id": "en_crea_story", "lang": "en", "category": "creative",
     "text": "Write the opening line of a detective novel set on a submarine."},
    {"id": "en_crea_invent", "lang": "en", "category": "creative",
     "text": "Invent a new holiday and describe how people celebrate it."},
    {"id": "en_open_advice", "lang": "en", "category": "open",
     "text": "What advice would you give to someone starting their first job?"},
    {"id": "en_open_future", "lang": "en", "category": "open",
     "text": "How do you imagine education will change in the next fifty years?"},
    # --- English, ambiguous ---
    {"id": "en_amb_bank", "lang": "en", "category": "ambiguous",
     "text": "I sat on the bank. Where do you think I am?"},
    {"id": "en_amb_bat", "lang": "en", "category": "ambiguous",
     "text": "Explain what a bat is, giving at least two different meanings."},
    {"id": "en_amb_lying", "lang": "en", "category": "ambiguous",
     "text": "Is it acceptable to lie in order to protect someone's feelings?"},
    {"id": "en_amb_chicken", "lang": "en", "category": "ambiguous",
     "text": "Why did the chicken cross the road?"},
]
