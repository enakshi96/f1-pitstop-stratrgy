import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY1"))


def generate_insights(strategy_data):
    
    prompt = f"""
    You are a Formula 1 race strategist. Based on the pit stop data below, analyze the race strategy in technical detail. Cover the following:
    1. Explain the likely reason behind each compound choice and its placement in the stint order.
    2. Discuss the relative length of each stint was it aggressive (short), balanced, or extended?
    3. Point out any unusual decisions (e.g., very early stops, short final stints, switching back to a softer compound, etc.)
    4. Comment on whether the strategy seems planned, reactive to race conditions, or opportunistic.
    5. Based on pit stop laps, infer if undercut or overcut was possibly attempted.
    6. Suggest what this strategy tells us about tire degradation or performance priorities.

    Strategy Type: {strategy_data['strategy']}
    Pit Stops: {strategy_data['pit_stops']}
    Stints: {strategy_data['stints']}
    """
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY1}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are an F1 strategy engineer providing detailed technical analysis."},
            {"role": "user", "content": prompt}
        ],
        "model": "mixtral-8x7b-32768"
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']