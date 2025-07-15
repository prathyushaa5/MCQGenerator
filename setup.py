from setuptools import find_packages, setup


setup(
    name="mcqgen",
    version="0.1.0",
    author="Prathyusha",
    author_email="prathyushaacharya050@gmail.com",
    description="A tool to generate multiple choice questions from text",
    
    install_requires=["openai", "langchain", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages(),
)