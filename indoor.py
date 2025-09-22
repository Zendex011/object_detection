# relationships.py

def detect_relationships(detections, vertical_threshold=0.1, horizontal_threshold=0.1):
    relationships = []
    
    # Precompute geometry for all objects
    for obj in detections:
        obj['center_x'] = (obj['x_min'] + obj['x_max']) / 2
        obj['center_y'] = (obj['y_min'] + obj['y_max']) / 2
        obj['width'] = obj['x_max'] - obj['x_min']
        obj['height'] = obj['y_max'] - obj['y_min']
    
    for i, objA in enumerate(detections):
        for j, objB in enumerate(detections):
            if i == j:
                continue
            
            A_bottom, A_top = objA['y_max'], objA['y_min']
            B_bottom, B_top = objB['y_max'], objB['y_min']
            
            vertical_gap_top = abs(B_top - A_bottom)
            vertical_gap_bottom = abs(A_top - B_bottom)
            
            # --- ON RELATIONSHIP ---
            if objB['x_min'] <= objA['center_x'] <= objB['x_max']:
                if vertical_gap_top < vertical_threshold * objB['height']:
                    relationships.append(f"{objA['label']} ON {objB['label']}")
            
            # --- UNDER RELATIONSHIP ---
            if objB['x_min'] <= objA['center_x'] <= objB['x_max']:
                if vertical_gap_bottom < vertical_threshold * objB['height']:
                    relationships.append(f"{objA['label']} UNDER {objB['label']}")
            
            # --- LEFT/RIGHT RELATIONSHIP ---
            horizontal_distance = abs(objA['center_x'] - objB['center_x'])
            avg_width = (objA['width'] + objB['width']) / 2
            if horizontal_distance > horizontal_threshold * avg_width:
                if objA['center_x'] < objB['center_x']:
                    relationships.append(f"{objA['label']} LEFT OF {objB['label']}")
                else:
                    relationships.append(f"{objA['label']} RIGHT OF {objB['label']}")
    
    return relationships
