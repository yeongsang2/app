import redis
import json

redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

def get_previous_searched_clothes_in_redis(tag) -> list:

    if redis_client.exists(tag):
        data = redis_client.lrange(tag, 0, -1)
        result_list = [json.loads(item) for item in data]
        return result_list

    return []


def add_clothes_to_list(tag, result):
        # Redis에 저장할 데이터를 JSON 형태로 직렬화
    result_json = json.dumps(result, ensure_ascii=False).encode('utf-8')

    # Redis에 데이터 추가
    if redis_client.exists(tag):
        # 기존 데이터에 추가
        redis_client.rpush(tag, result_json)
    else:
        # 새로운 리스트 생성
        redis_client.rpush(tag, result_json)
        # 만료 시간 설정 (5분)
        redis_client.expire(tag, 300)

