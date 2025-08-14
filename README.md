# 미니 프로젝트 : 교통법규 위반 신고 간편화 프로그램

## 목차

1. 개발 환경
   - 하드웨어
   - 개발 환경 및 사용 기술

2. 주제 선정
   - 프로젝트 주제
   - 관련 제품 링크
   - 기존 제품의 한계점과 개선점

## 1. 개발 환경

<details>
<summary></summary>
<div markdown="1">

## **1-1. 하드웨어**

개인 노트북
- GPU : NVIDIA GeForce 940MX

웹캠
- Logitech C920

## **1-2. 개발 환경 및 사용 기술**

| 언어 | 라이브러리 / 프레임워크 | 개발 도구 | 운영체제 |
|------|------------------------|-----------|-----------|
| ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) | ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv&logoColor=white) ![YOLO](https://img.shields.io/badge/YOLO-v11-orange) | ![VSCode](https://img.shields.io/badge/VSCode-blueviolet?logo=visual-studio-code&logoColor=white) | ![Windows](https://img.shields.io/badge/Windows-10-lightgrey?logo=windows&logoColor=white) |

</div>
</details>

## 2. 주제 선정

<details>
<summary></summary>
<div markdown="1">

## **2-1. 프로젝트 주제**

**주행 중 교통법규 위반 이벤트 발생시 자동으로 영상 캡쳐 후 지정된 위치로 영상 발송**

## **2-2. 관련 제품 링크**

_국내 블랙박스 브랜드 점유율 TOP 2 기업_

[아이나비](https://www.inavi.com/)

[파인뷰](http://www.fine-drive.com/defaults/index.do)

## **2-3. 기존 제품의 한계점과 개선점**

**[1. 한계점]**

기존의 블랙박스는 이벤트 발생(교통법규 위반)시 **임의로 제품이나 차량에 충격을 가하여 이벤트 순간을 특정**하거나,

주행이 끝난 뒤 스스로 이벤트 발생 순간을 확인해야함.

**[2. 개선점]**

이벤트 발생 순간을 **openCV와 YOLO를 통해 자동으로 인식**하고, 해당 부분의 영상을 캡쳐하여 지정한 위치(메일, 공유 폴더 등)로 자동으로 전송함.

</div>
</details>