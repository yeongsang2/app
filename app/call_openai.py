from dotenv import load_dotenv
import tiktoken as tiktoken
import openai
import os
import time
import json


# Load default environment variables (.env)
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


openai.api_key = OPENAI_API_KEY


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    system_prompt: str,
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system" , "content": system_prompt},{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break




def request_openai(previous_searched_clothes: list(), result):

    system_prompt = """
        You are the Fashion coordinator for the visually impaired. 
        """
    current_outfit = "logo : {logo}, color : {color}, pattern : {pattern}, type: {type}".format(logo=result['logo'], color=result['color'], pattern=result['pattern'], type=result['type'])
    if(previous_searched_clothes):
        previous_clothes = ''
        for i, p in enumerate(previous_searched_clothes):
            logo, color, pattern, type = p['logo'], p['color'], p['pattern'], p['type']
            previous_clothes  += "{i} - logo : {logo}, color : {color}, pattern : {pattern}, type: {type}".format(i = str(i), logo=logo, color=color, pattern=pattern, type=type)
    else:
        previous_clothes = 'none'
    prompt = f"""
        Based on the clothes you previously searched for, 
        recommend outfits that go well with the current outfit you're looking at. 
        If there are no previous clothing searches or if there are no outfits that complement the current one, 
        recommend based on your knowledge of clothing. 
        Recommend tops to match bottoms, and bottoms to match tops.
        Please provide information about the current outfit before suggesting complementary clothing.
    
        Example:
            Previous Clothing Searches:
              1 - logo: 나이키 , color: 흰색,  pattern: 줄무늬, type: 반바지,
              2 - logo: 없음 , color: 보라색, pattern: 민무늬, type: 반바지

            Current Outfit:
              `logo: 나이키, color: 검은색, pattern: 줄무늬, type: 반팔`

            Output Example:
             `지금 조회하신 의상은 검정색 나이키 스트라이프 반팔입니다. 이전에 조회했던 흰색 나이키 스트라이프 반바지랑 잘 어울릴 것 같아요. 이 두 의상을 함께 입으면 스포티하고 시원한 룩을 완성할 수 있을 것입니다.`

        Previous Clothing Searches:   
            {previous_clothes}   
        Current Outfit:
            {current_outfit}
        You need to comply with the following requirements
        requirements:
            - This translation assumes that you want the given text translated into Korean
            - Make sure the output is less than 100 words
        """
    try:
        response = openai_call(system_prompt, prompt, max_tokens=2000)
        return response
    except:
        print("error response")
