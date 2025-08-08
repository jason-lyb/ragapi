import requests
import json
from config import KAKAO_REST_API_KEY

def search_places_by_keyword(keyword, x=None, y=None, radius=20000, page=1, size=15):
    """
    키워드로 장소를 검색하는 함수
    
    Args:
        keyword (str): 검색 키워드
        x (float, optional): 중심 좌표의 X 혹은 longitude (경도)
        y (float, optional): 중심 좌표의 Y 혹은 latitude (위도)
        radius (int): 중심 좌표부터의 반경거리(미터), 기본값 20000m
        page (int): 결과 페이지 번호, 기본값 1
        size (int): 한 페이지에 보여질 문서의 개수, 기본값 15
    
    Returns:
        dict: 검색 결과
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    
    params = {
        "query": keyword,
        "page": page,
        "size": size
    }
    
    # 좌표가 제공된 경우 추가
    if x is not None and y is not None:
        params["x"] = x
        params["y"] = y
        params["radius"] = radius
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # 결과를 보기 좋게 정리
        places = []
        for document in data.get('documents', []):
            place_info = {
                'place_name': document.get('place_name'),
                'category_name': document.get('category_name'),
                'address_name': document.get('address_name'),  # 전체 지번 주소
                'road_address_name': document.get('road_address_name'),  # 전체 도로명 주소
                'phone': document.get('phone'),
                'place_url': document.get('place_url'),
                'x': document.get('x'),  # 경도
                'y': document.get('y'),  # 위도
                'distance': document.get('distance')  # 중심좌표까지의 거리 (단위: meter)
            }
            
            # 도로명 주소가 있으면 도로명, 없으면 지번 주소 표시 + 장소명 추가
            if place_info['road_address_name']:
                place_info['display_address'] = f"{place_info['road_address_name']} ({place_info['place_name']})"
            else:
                place_info['display_address'] = f"{place_info['address_name']} ({place_info['place_name']})"
                
            places.append(place_info)
        
        return {
            'places': places,
            'meta': data.get('meta', {}),
            'total_count': data.get('meta', {}).get('total_count', 0)
        }
        
    except requests.exceptions.RequestException as e:
        return {'error': f'API 요청 중 오류가 발생했습니다: {str(e)}'}
    except json.JSONDecodeError as e:
        return {'error': f'응답 데이터 파싱 중 오류가 발생했습니다: {str(e)}'}


def coord_to_address(x, y):
    """
    좌표를 주소로 변환하는 함수
    
    Args:
        x (float): 경도 (longitude)
        y (float): 위도 (latitude)
    
    Returns:
        dict: 주소 정보
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }
    
    params = {
        "x": x,
        "y": y,
        "input_coord": "WGS84"  # 좌표계 (WGS84, WCONGNAMUL, CONGNAMUL, WTM, TM)
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('documents'):
            return {'error': '해당 좌표에 대한 주소를 찾을 수 없습니다.'}
        
        document = data['documents'][0]
        
        # 도로명 주소와 지번 주소 정보 추출
        road_address = document.get('road_address')
        address = document.get('address')
        
        result = {
            'x': x,
            'y': y
        }
        
        # 도로명 주소 정보
        if road_address:
            result['road_address'] = {
                'address_name': road_address.get('address_name'),
                'region_1depth_name': road_address.get('region_1depth_name'),  # 시도
                'region_2depth_name': road_address.get('region_2depth_name'),  # 시군구
                'region_3depth_name': road_address.get('region_3depth_name'),  # 읍면동
                'road_name': road_address.get('road_name'),
                'underground_yn': road_address.get('underground_yn'),
                'main_building_no': road_address.get('main_building_no'),
                'sub_building_no': road_address.get('sub_building_no'),
                'building_name': road_address.get('building_name'),
                'zone_no': road_address.get('zone_no')
            }
        
        # 지번 주소 정보
        if address:
            result['jibun_address'] = {
                'address_name': address.get('address_name'),
                'region_1depth_name': address.get('region_1depth_name'),  # 시도
                'region_2depth_name': address.get('region_2depth_name'),  # 시군구
                'region_3depth_name': address.get('region_3depth_name'),  # 읍면동
                'mountain_yn': address.get('mountain_yn'),
                'main_address_no': address.get('main_address_no'),
                'sub_address_no': address.get('sub_address_no')
            }
        
        # 도로명 주소가 있으면 도로명, 없으면 지번 주소를 표시용으로 설정
        # 좌표->주소 변환에서는 장소명이 없으므로 주소만 표시
        if road_address and road_address.get('address_name'):
            display_addr = road_address['address_name']
            # 동명과 건물명이 있으면 추가
            additional_info = []
            if road_address.get('region_3depth_name'):
                additional_info.append(road_address['region_3depth_name'])
            if road_address.get('building_name'):
                additional_info.append(road_address['building_name'])
            
            if additional_info:
                display_addr += f" ({', '.join(additional_info)})"
            
            result['display_address'] = display_addr
            result['address_type'] = '도로명'
        elif address and address.get('address_name'):
            result['display_address'] = address['address_name']
            result['address_type'] = '지번'
        else:
            result['display_address'] = '주소를 찾을 수 없습니다'
            result['address_type'] = 'unknown'
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {'error': f'API 요청 중 오류가 발생했습니다: {str(e)}'}
    except json.JSONDecodeError as e:
        return {'error': f'응답 데이터 파싱 중 오류가 발생했습니다: {str(e)}'}


# 사용 예시
if __name__ == "__main__":
    # API 키를 실제 키로 변경해주세요
    KAKAO_API_KEY = "YOUR_ACTUAL_KAKAO_REST_API_KEY"
    
    print("=== 키워드로 장소 검색 예시 ===")
    # 키워드로 장소 검색
    result = search_places_by_keyword("카카오")
    if 'error' not in result:
        print(f"총 {result['total_count']}개의 장소를 찾았습니다.")
        for i, place in enumerate(result['places'][:5], 1):  # 상위 5개만 표시
            print(f"{i}. {place['place_name']}")
            print(f"   주소: {place['display_address']}")
            print(f"   카테고리: {place['category_name']}")
            print(f"   전화번호: {place['phone']}")
            print(f"   좌표: ({place['x']}, {place['y']})")
            print()
    else:
        print(f"오류: {result['error']}")
    
    print("\n=== 좌표를 주소로 변환 예시 ===")
    # 좌표를 주소로 변환 (카카오 본사 좌표)
    address_result = coord_to_address(127.108678, 37.402001)
    if 'error' not in address_result:
        print(f"좌표 ({address_result['x']}, {address_result['y']})의 주소:")
        print(f"표시 주소 ({address_result['address_type']}): {address_result['display_address']}")
        
        if 'road_address' in address_result:
            print(f"도로명 주소: {address_result['road_address']['address_name']}")
        
        if 'jibun_address' in address_result:
            print(f"지번 주소: {address_result['jibun_address']['address_name']}")
    else:
        print(f"오류: {address_result['error']}")