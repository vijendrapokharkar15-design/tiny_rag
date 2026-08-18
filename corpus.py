# small test corpus for the tiny RAG
# 4 short passages, all about Mars missions, so retrieval has to work on meaning

docs = [
    "The Perseverance rover landed in Jezero Crater in February 2021. Its main goal is to look for signs of ancient microbial life and to collect rock samples for a future return mission.",
    "Perseverance carries a small helicopter called Ingenuity. Ingenuity made the first powered flight on another planet in April 2021 and went on to fly dozens of times before its rotor was damaged.",
    "The Curiosity rover landed in Gale Crater in 2012 and is still operating. It found chemical evidence that Mars once had long-lived lakes of liquid water.",
    "Mars has a very thin atmosphere, about one percent the density of Earth's, and it is mostly carbon dioxide. This makes powered flight difficult and offers little protection from radiation.",
]

# test questions - I know which doc should answer each one

questions = [
    "Which mission flew a helicopter on Mars?",        # expect doc 1
    "What did Curiosity discover about water?",        # expect doc 2
    "Why is flying on Mars hard?",                     # expect doc 3
]


# questions the corpus cannot answer - used to calibrate a score threshold

out_of_scope = [
    "Who was the first person to walk on the Moon?",
    "What is the capital of France?",
    "How do I train a neural network?",
    "What is the best way to make sourdough bread?",
]