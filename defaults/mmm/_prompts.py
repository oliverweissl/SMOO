DETECTION_PROMPT = (
    'Detect the object(s) "{objects}" in this image. '
    "Return ONLY a JSON array — no markdown, no commentary. "
    "Each element must have exactly two keys: "
    '"label" (string, one of the requested objects) and '
    '"bbox" ([x1, y1, x2, y2] bounding box coordinates). '
    'Example: [{{"label": "cat", "bbox": [10, 20, 150, 200]}}, '
    '{{"label": "dog", "bbox": [300, 50, 480, 400]}}]'
)
