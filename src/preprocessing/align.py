

# -----------------------------
# READ ALIGN FILE
# -----------------------------
def read_align_file(align_path):
    """
    Parses alignment files to extract word-level time segments, excluding silence.
    """
    segments = []
    with open(align_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            
            start, end, word = parts
            if word in ['sil', 'sp']:
                continue  # skip silence & short pause
            
            start = int(start)
            end = int(end)
            
            segments.append((start, end, word))
    
    return segments
