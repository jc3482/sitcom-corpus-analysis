from .mbti_prediction import (
    load_bundle,
    load_all_dialogues,
    predict_mbti_for_character,
    score_quotes_for_character,
    SHOW_FILES,
    SHOW_DISPLAY_NAMES,
    SHOW_ALIASES,
)


def main():
    print("[INFO] Loading MBTI model bundle...")
    tokenizer, model, label_mapping, preproc_info, max_len, device = load_bundle()

    print("[INFO] Loading and preprocessing TV scripts...")
    df_all = load_all_dialogues()

    print("\n=== MBTI Character Explorer ===")
    print("Available shows:")
    for key in SHOW_FILES.keys():
        print(f"  - {SHOW_DISPLAY_NAMES.get(key, key)}  (input key: {key})")
    print("\nExamples: friends, modern family, seinfeld, tbbt, the office")
    print("Type 'quit' to exit.\n")

    while True:
        show_inp = input("Enter show name (or 'quit'): ").strip().lower()
        if show_inp in ("quit", "exit", "q"):
            print("Process terminated.")
            break

        show_key = SHOW_ALIASES.get(show_inp, None)
        if show_key is None:
            if show_inp in SHOW_FILES:
                show_key = show_inp
            else:
                print("[WARN] Unknown show. Try: friends / modern family / seinfeld / tbbt / the office")
                continue

        show_name_pretty = SHOW_DISPLAY_NAMES.get(show_key, show_key)
        print(f"Selected show: {show_name_pretty}")

        char_inp = input("Enter character name (e.g. 'Ross Geller'): ").strip()
        if not char_inp:
            print("[WARN] Character name cannot be empty.")
            continue

        mbti_str, dim_probs, df_char = predict_mbti_for_character(
            tokenizer, model, label_mapping, df_all, show_key, char_inp
        )

        if df_char is None or df_char.empty:
            print(f"[WARN] No dialogue found for '{char_inp}' in {show_name_pretty}.")
            continue

        if mbti_str is None:
            print(f"[WARN] Could not generate MBTI for '{char_inp}'.")
            continue

        print(f"\n=== Result for {char_inp} ({show_name_pretty}) ===")
        print(f"Predicted MBTI: {mbti_str}")

        print("\nPer-dimension probabilities:")
        for dim, probs in dim_probs.items():
            print(f"  {dim}: {probs}")

        top_quotes = score_quotes_for_character(
            tokenizer, model, label_mapping, df_char, mbti_str, top_k=5
        )

        if not top_quotes:
            print("\n[INFO] No strong quotes found (maybe all lines are very short).")
        else:
            print("\nTop quotes that best reflect this MBTI:")
            for i, q in enumerate(top_quotes, start=1):
                print(f"\n[{i}] S{q['season']}E{q['episode']} - {q['episode_title']}")
                print(f"Score: {q['score']:.4f}")
                print(f"Quote: {q['dialogue_raw']}")

        print("\n---------------------------------------------\n")


if __name__ == "__main__":
    main()
