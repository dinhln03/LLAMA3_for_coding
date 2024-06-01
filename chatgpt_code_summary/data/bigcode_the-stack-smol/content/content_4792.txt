'''
file.readline() 사용해서 csv 파일 열기
'''
#
# def my_csv_reader(fn:str, header=True) -> list:
#     '''
#     csv 파일의 데이터 2차원 행렬 형태로 리턴
#
#
#     :param fn: 읽을 파일 이름(예: data\\exam.csv)
#     :param header: csv파일의 헤더 존재 여부
#     :return: csv 파일에서 헤더는 제외한 데이터로 이루어진 2차원 리스트
#     '''
#
#
# if __name__ == '__main__':
#
#     # 작성한 함수들을 테스트
#     pass

def print_data(data: list) -> None:
    '''
    2차원 리스트의 내용을 출력
    1 10 20 30 40
    2 11 21 31 41
    ...


    :param data: 2차원 행렬 형태의 리스트
    :return: None
    '''

    readcsv = open('data/exam.csv', mode ='r', encoding='utf-8')
    line=readcsv.readline()
    while line:
        print(line.strip())
        line = readcsv.readline()

    readcsv.close()


# def get_sum_mean(data : list, col: int) -> tuple:
#     '''
#     주어진 2차원 리스트(data)에서 해당 컬럼(col)의 데이터들의
#     총합(sum)과 평균(mean)을 계산해서 리턴
#
#     :param data: 2차원 행렬 형태의 리스트
#     :param col: 컬럼 인덱스(0,1,2,...)
#     :return: 컬럼 데이터의 합과 평균
#     '''