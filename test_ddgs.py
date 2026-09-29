from duckduckgo_search import DDGS
prompt = 'Return exactly this JSON: {\"Campaign Objective\": \"Test\", \"Psychological Angle\": \"Test\", \"Persona\": \"Test\", \"Storyline\": \"Test\"}'
try:
    res = DDGS().chat(prompt, model='gpt-4o-mini')
    print('Response:', res)
except Exception as e:
    print('Error:', e)
