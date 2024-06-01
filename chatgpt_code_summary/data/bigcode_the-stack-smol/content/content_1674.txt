from setuptools import setup
import mp_sync

setup(
    name='mp_sync',
    version=mp_sync.__version__,
    description='Moon Package for Sync repository(google drive, notion, mongodb(local/web), local file)',
    url='https://github.com/hopelife/mp_sync',
    author='Moon Jung Sam',
    author_email='monblue@snu.ac.kr',
    license='MIT',
    packages=['mp_sync'],
    # entry_points={'console_scripts': ['mp_sync = mp_sync.__main__:main']},
    keywords='scraper',
    # python_requires='>=3.8',  # Python 3.8.6-32 bit
    # install_requires=[ # 패키지 사용을 위해 필요한 추가 설치 패키지
    #     'selenium',
    # ],
    # zip_safe=False
)
