# JadenCase 문자열 만들기

def solution(s):
    s = s.lower()
    changed_words = []
    print(s.split(" "))
    for word in s.split(" "):
        if len(word) == 0:
            changed_words.append(word)
            continue
        elif len(word) == 1:
            word = word[0].upper()
        else:
            word = word[0].upper() + word[1:]
        changed_words.append(word)
    print(changed_words)
    answer = ' '.join(changed_words)
    return answer
    
'''
채점을 시작합니다.
정확성  테스트
테스트 1 〉	통과 (0.02ms, 10.3MB)
테스트 2 〉	통과 (0.02ms, 10.1MB)
테스트 3 〉	통과 (0.02ms, 10.2MB)
테스트 4 〉	통과 (0.02ms, 10.1MB)
테스트 5 〉	통과 (0.03ms, 10.2MB)
테스트 6 〉	통과 (0.02ms, 10.1MB)
테스트 7 〉	통과 (0.03ms, 10.2MB)
테스트 8 〉	통과 (0.01ms, 10.2MB)
테스트 9 〉	통과 (0.02ms, 10.2MB)
테스트 10 〉	통과 (0.01ms, 10.1MB)
테스트 11 〉	통과 (0.03ms, 10.2MB)
테스트 12 〉	통과 (0.02ms, 10.2MB)
테스트 13 〉	통과 (0.02ms, 10.2MB)
테스트 14 〉	통과 (0.02ms, 10.2MB)
테스트 15 〉	통과 (0.03ms, 10.2MB)
테스트 16 〉	통과 (0.01ms, 10.2MB)
채점 결과
정확성: 100.0
합계: 100.0 / 100.0
'''
