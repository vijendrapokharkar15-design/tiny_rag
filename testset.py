# labelled test set for retrieval evaluation
# each entry: the question, and the doc index(es) that answer it
# labels found by keyword search with grep_corpus.py - NOT by the embedding model

TESTSET = [
    {
        "q": "What is the presumption in favour of sustainable development?",
        "docs": [116],
    },
    {
        "q": "What counts as affordable housing?",
        "docs": [444],
    },
    {
        "q": "Can affordable housing be provided off-site instead of on the development site?",
        "docs": [156],
    },
    {
        "q": "What contributions are required for housing on land released from the Green Belt?",
        "docs": [263, 265],
    },
    {
        "q": "Which types of building are classed as highly vulnerable to flooding?",
        "docs": [603],
    },
    {
        "q": "What design standards apply to essential infrastructure in the functional floodplain?",
        "docs": [613],
    },
    {
        "q": "What is a heritage asset?",
        "docs": [487],
    },
    {
        "q": "What happens to archaeological remains that cannot be preserved on site?",
        "docs": [430],
    },
]