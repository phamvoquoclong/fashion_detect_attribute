# utils/attribute_mapper.py
def load_attribute_map(txt_path):
    attr_id2name = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                attr_id2name[int(parts[0])] = parts[1]

    return attr_id2name
