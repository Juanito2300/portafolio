def detect_profile(client, message):

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Clasifica el usuario. Responde SOLO: DEV, RECLUTADOR u OTRO."
                },
                {"role": "user", "content": message}
            ]
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print("Error detector:", e)
        return "OTRO"