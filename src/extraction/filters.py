from sentence_transformers import util

TECHNICAL_ANCHORS = [
    "programming language skill",
    "machine learning algorithm",
    "data analysis and visualization tool",
    "database management system",
    "software development framework",
    "computer science fundamental",
    "cloud infrastructure platform",
    "API design and integration",
    "statistical modeling technique",
    "version control system",
    "deep learning neural network",
    "software engineering methodology",
]

ADMINISTRATIVE_ANCHORS = [
    "employee benefit and compensation",
    "work schedule and time off",
    "office location and remote work policy",
    "salary range and pay",
    "health insurance and medical coverage",
    "employment terms and conditions",
    "company policy and compliance",
    "job perks and incentives",
    "equal opportunity employment statement",
    "background check requirement",
]

_anchor_cache: dict = {}


def get_anchor_embeddings(sbert_model) -> tuple:
    key = id(sbert_model)
    if key not in _anchor_cache:
        tech_emb = sbert_model.encode(TECHNICAL_ANCHORS, convert_to_tensor=True)
        admin_emb = sbert_model.encode(ADMINISTRATIVE_ANCHORS, convert_to_tensor=True)
        _anchor_cache[key] = (tech_emb, admin_emb)
    return _anchor_cache[key]


def semantic_relevance_filter(
    entities: set,
    sbert_model,
    tech_threshold: float = 0.30,
) -> set:
    if not entities:
        return set()

    entity_list = list(entities)
    entity_embeddings = sbert_model.encode(entity_list, convert_to_tensor=True)

    tech_emb, admin_emb = get_anchor_embeddings(sbert_model)

    tech_sim = util.cos_sim(entity_embeddings, tech_emb)
    admin_sim = util.cos_sim(entity_embeddings, admin_emb)

    max_tech = tech_sim.max(dim=1).values
    max_admin = admin_sim.max(dim=1).values

    filtered = set()
    for i, entity in enumerate(entity_list):
        if max_tech[i].item() >= tech_threshold and max_tech[i].item() > max_admin[i].item():
            filtered.add(entity)

    return filtered
