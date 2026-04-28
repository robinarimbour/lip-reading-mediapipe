

def read_align_file(align_path):
    """
    Parses alignment files to extract word-level time segments,
    excluding silence tokens.

    Returns:
        List of tuples: (start_time, end_time, word)
    """
    segments = []

    try:
        with open(align_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 3:
                    continue
                
                start, end, word = parts

                # skip silence & short pause
                if word in ['sil', 'sp']:
                    continue
                
                start = int(start)
                end = int(end)
                
                segments.append((start, end, word))
    
    except FileNotFoundError:
        print(f"[ERROR] Align file not found: {align_path}")

    except Exception as e:
        print(f"[ERROR] Failed to read {align_path}: {e}")

    return segments
