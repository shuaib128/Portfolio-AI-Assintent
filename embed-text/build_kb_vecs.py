import json, httpx, os, time, sys, subprocess

# Make sure folders exist
os.makedirs("mp3", exist_ok=True)

# Load your knowledge base
with open("kb.json") as f:
    kb = json.load(f)

kb_with_vecs = []
total = len(kb)

print(f"\n🚀 Starting build for {total} entries...\n")

for i, item in enumerate(kb, 1):
    print(f"({i}/{total}) 🔹 Processing: {item['id']} ...", end="", flush=True)

    # --- Step 1: Get embedding ---
    try:
        emb_response = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": item["q"]},
            timeout=None
        )
        vec = emb_response.json()["embedding"]
    except Exception as e:
        print(f"\n⚠️ Failed embedding for {item['id']}: {e}")
        continue

    # --- Step 2: Generate audio using Coqui-TTS ---
    wav_path = f"mp3/{item['id']}.wav"
    mp3_path = f"mp3/{item['id']}.mp3"
    tts_params = {
        "text": item["a"],
        "length_scale": 2,
    }

    for attempt in range(3):
        try:
            start = time.time()
            tts_response = httpx.get(
                "http://localhost:5002/api/tts",
                params=tts_params,
                timeout=None
            )
            duration = time.time() - start

            # Save audio to WAV
            with open(wav_path, "wb") as f:
                f.write(tts_response.content)

            # Convert WAV → MP3 (quality level 3 ≈ 192 kbps)
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "3", mp3_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            os.remove(wav_path)

            print(f"\r({i}/{total}) ✅ {item['id']} → {mp3_path} ({duration:.1f}s)")
            break

        except Exception as e:
            print(f"\n⚠️ TTS failed ({attempt+1}/3) for {item['id']}: {e}")
            time.sleep(5)
    else:
        print(f"\n❌ Skipping {item['id']} after 3 failed attempts.")
        continue

    # --- Step 3: Append with embedding and MP3 path ---
    kb_with_vecs.append({
        **item,
        "vec": vec,
        "audio": mp3_path
    })

# --- Step 4: Save results ---
with open("kb_embedded.json", "w") as f:
    json.dump(kb_with_vecs, f, indent=2)

print("\n🎉 All entries processed, converted to MP3, and saved to kb_embedded.json")
