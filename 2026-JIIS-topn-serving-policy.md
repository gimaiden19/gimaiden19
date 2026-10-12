# 결측 영양 정보 환경에서 제약 기반 레시피 추천의 Top-N 반환 보장을 위한 운영 정책: 불확실성 정량화 및 제약충족 가능성 정렬

*An Operating Policy for Guaranteeing Top-N Return in Constraint-Based Recipe Recommendation under Missing Nutrition Information: Uncertainty Quantification and Constraint-Satisfaction Ranking*

| | |
|---|---|
| **Authors** | 김종훈 (Jonghun Gim), 고선우 (Sunwoo Ko) — 전주대학교 인공지능학과 |
| **Venue** | 지능정보연구 (Journal of Intelligence and Information Systems), Vol. 32, No. 1, pp. 351–379, March 2026 |
| **Status** | **Published** (KCI) |
| **DOI** | [10.13088/jiis.2026.32.1.351](https://doi.org/10.13088/jiis.2026.32.1.351) |
| **Language** | Korean |
| **Role** | First author — problem formulation, policy design, experiments, writing |
| **Related** | Follow-up: [NutriHeal (KBS, under review)](2026-KBS-nutriheal.md) · Patents [P1, P2](patents.md) |

## 한 줄 요약

영양 정보가 빠진 레시피를 **"확실히 만족 / 확실히 위반 / 판단 불가"** 세 상태로 나누고, 판단 불가 항목을 숨기지 않고 표시하면서도 요청한 개수(Top-N)를 끝까지 채우는 추천 서빙 정책.

## 문제

제약 기반 추천(예: 나트륨·열량 상한)에서는 후보가 조건을 만족하는지 확인해야 하는데, 실제 데이터는 영양 값이 비어 있거나, 재료가 영양 DB에 연결되지 않거나(매핑 실패), 연결 신뢰도가 낮거나, 중량이 불완전합니다. 이때 시스템은 두 방향으로 실패합니다.

- **반환 붕괴(collapse)** — 확실히 만족하는 후보만 보여주면 안전해 보이지만, 조건이 엄격해질수록 요청한 개수를 채우지 못함
- **위험 은폐(concealment)** — 개수를 맞추려고 불확실한 후보를 섞어 넣으면, 사용자에게 그 사실이 드러나지 않음

결측을 "메워서 없애는" 문제가 아니라 **"정보가 부족한 상태에서 추천을 어떻게 운영할 것인가"**의 문제로 다시 정의했습니다.

## 방법

1. 후보를 **strict**(현재 정보만으로 만족), **violate**(현재 정보만으로 위반), **review**(판단 불가)의 3상태로 분류
2. strict를 우선 제시하고 부족분을 review로 채우되, 결과에 **불확실 항목의 포함 비율을 함께 표시**하는 서빙 정책
3. review 항목 사이에서도 제약을 만족할 **가능성이 높은 순으로 정렬**하는 확장안

## 결과

- 제약이 강해지거나 금지 재료 같은 추가 제한이 붙어도 요청 개수의 추천을 더 안정적으로 반환
- 불확실성이 어디서 어떻게 늘어나는지 해석 가능하게 노출하여 운영 투명성 향상

## 왜 이 논문이 다른 연구와 연결되는가

"모델이 확신하지 못하는 판단을 **보류로 노출한다**"는 원칙은 이후 [NutriHeal](2026-KBS-nutriheal.md)의 선별 검증(selective verification)으로 발전했고, 제조 AI 에이전트의 **승인 게이트**(자동 실행 / 사람 확인 분리) 설계에 그대로 적용할 수 있는 원칙이라고 보고 있습니다.

## Keywords

제약 기반 추천 · 레시피 추천 · 결측 데이터 · 불확실성 노출 · Top-N 반환 보장

## Cite

```bibtex
@article{gim2026topn,
  author  = {Gim, Jonghun and Ko, Sunwoo},
  title   = {An Operating Policy for Guaranteeing Top-N Return in Constraint-Based Recipe Recommendation under Missing Nutrition Information: Uncertainty Quantification and Constraint-Satisfaction Ranking},
  journal = {Journal of Intelligence and Information Systems},
  volume  = {32}, number = {1}, pages = {351--379}, year = {2026},
  doi     = {10.13088/jiis.2026.32.1.351},
  note    = {in Korean}
}
```
