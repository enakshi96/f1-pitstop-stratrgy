import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_llm_explanation(input_data: dict, prediction: list) -> str:
    strategy_type = prediction[0]
    compounds = prediction[1:4]
    stint_lengths = prediction[4:7]

    user_prompt = f"""
    You are an F1 strategy engineer AI.

    A pit stop strategy has been predicted using machine learning. Here's the input:

    - Qualifying Position: {input_data["QualiPos"]}
    - Average Air Temperature: {input_data["AvgAirTemp"]}°C
    - Rain: {'Yes' if input_data["Rain"] == 1 else 'No'}
    - Pit Loss Time: {input_data["PitLossTime"]} seconds
    - Is Street Circuit: {'Yes' if input_data["IsStreetCircuit"] == 1 else 'No'}
    - Track Length: {input_data["TrackLength_km"]} km

    Prediction:
    - Strategy Type: {strategy_type}
    - Compounds: {compounds}
    - Stint Lengths: {stint_lengths} laps

    Explain in clear, simple terms why this strategy might have been chosen.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
              messages=[
                  {"role": "system", "content": "You are an expert F1 race strategist."},
                  {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
                )
        return response.choices[0].message.content.strip()
    except Exception as e:
         return f"⚠️ LLM explanation unavailable: {str(e)}"