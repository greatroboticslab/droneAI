from pathlib import Path
import traceback

import tartanair as ta


def main():
    root = Path(__file__).resolve().parent / "TartanAirV2Data"
    root.mkdir(parents=True, exist_ok=True)

    print("Initializing TartanAir V2 at:")
    print(root)

    ta.init(str(root))

    print("\nTartanAir module loaded successfully.")
    print("Available top-level functions:")
    names = [name for name in dir(ta) if not name.startswith("_")]
    for name in names:
        print(" -", name)

    print("\nTrying to read available dataset metadata...")
    try:
        all_data = ta.get_all_data()
        print("get_all_data() worked.")
        print("Type:", type(all_data))

        if isinstance(all_data, dict):
            print("Keys:", list(all_data.keys()))
            for key, value in all_data.items():
                preview = value[:5] if isinstance(value, list) else value
                print(f"{key}: {preview}")
        else:
            print(all_data)

    except Exception:
        print("get_all_data() failed:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
