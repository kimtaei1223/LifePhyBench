# LifePhyBench 연구 설명 및 차별성 점검

> 선행연구 점검 기준일: 2026-08-04  
> 연구 단계: 개발 및 벤치마크 타당성 검증 단계  
> 목표 투고처: Transactions on Machine Learning Research (TMLR)

## 1. 한 문장 요약

**LifePhyBench는 에이전트의 행동이 로봇의 숨은 물리적 건강 상태를
변화시키고, 그 변화가 task episode가 끝난 뒤에도 남아 이후의 물리
동역학과 장기 성능에 영향을 주는 조건에서의 의사결정·적응·평가를
연구하는 시뮬레이션 벤치마크다.**

연구의 가제는 다음과 같다.

> **LifePhyBench: Decision-Making under Endogenous Persistent Physical
> Degradation and Selective Resets**

## 2. 연구 배경과 문제의식

일반적인 로봇 강화학습 벤치마크에서는 episode가 끝나면 다음 요소가
대부분 함께 초기화된다.

- 로봇과 물체의 위치
- 속도와 관절 상태
- actuator 성능
- 온도와 피로
- 마모 및 손상

하지만 실제 장비에서는 작업 위치는 다시 맞출 수 있어도 모터 마모,
열 축적, 관절 열화까지 새 제품 상태로 돌아가지는 않는다.

예를 들어 로봇이 현재 작업을 빨리 끝내기 위해 지속적으로 큰 토크를
사용하면 즉각적인 보상은 좋아질 수 있다. 그러나 그 행동이 actuator를
열화시키면 이후 10개 또는 20개 작업의 수행 능력은 나빠질 수 있다.
따라서 에이전트는 현재 작업만 수행하는 것이 아니라 다음 두 문제를
동시에 풀어야 한다.

1. 보이지 않는 물리적 건강 상태를 과거 관측과 행동으로 추론한다.
2. 현재 성과와 장기적인 장비 보존 사이의 균형을 계획한다.

이것이 일반적인 domain randomization이나 episode 단위 robust RL과
구분되는 핵심 문제다.

## 3. 문제의 수학적 표현

관측 가능한 작업 상태를 $x_t$, 숨은 물리적 건강 상태를 $z_t$, 행동을
$a_t$라고 두면 다음과 같이 표현할 수 있다.

$$
x_{t+1}\sim P_x(\cdot\mid x_t,z_t,a_t)
$$

$$
z_{t+1}\sim P_z(\cdot\mid z_t,x_t,a_t)
$$

여기서 중요한 조건은 다음과 같다.

- 행동 $a_t$가 현재 task state뿐 아니라 건강 상태 $z_t$도 변화시킨다.
- 건강 상태 $z_t$가 이후 task dynamics $P_x$에 영향을 준다.
- 일반적인 episode reset은 $x_t$만 초기화한다.
- lifetime reset만 $x_t$와 $z_t$를 모두 초기화한다.

따라서 두 종류의 reset을 명시적으로 구분한다.

| Reset | 작업 상태 | 물리적 건강 상태 |
|---|---|---|
| `reset_episode()` | 초기화 | 유지 또는 부분 회복 |
| `reset_lifetime()` | 초기화 | 완전 초기화 |

완전한 상태 $(x,z)$를 포함하면 이는 고정된 전이법칙을 가진 latent-state
POMDP로 볼 수 있다. 따라서 단순히 마모가 진행된다는 이유만으로
“수학적으로 비정상적인 환경”이라고 주장하지 않는다.

## 4. 현재 구현된 열화 메커니즘

현재 무료 MuJoCo 환경에서 다음 세 가지 물리 채널을 구현했다.

- **Actuator wear:** 행동량이 누적될수록 actuator gain 감소
- **Thermal derating:** 부하에 따라 열이 축적되고 episode 경계에서 일부 냉각
- **Joint aging:** 사용량에 따라 관절 damping 증가

Pusher-v5와 Reacher-v5에서 다음 조건을 독립적으로 조절할 수 있다.

- persistent health 대 episode-reset health
- endogenous degradation 대 exogenous clock drift
- hidden health 대 privileged health
- power, threshold, stochastic-shock 열화법칙

Endogenous 조건은 행동이나 접촉량에 따라 열화가 달라지는 조건이고,
exogenous 조건은 정책과 무관하게 시간에 따라 열화되는 대조군이다.

## 5. 선행연구와의 관계

### 5.1 행동이 미래 환경을 변화시키는 문제

Chandak 등의 연구는 active non-stationarity를 정의하고, 과거 행동이
로봇 모터의 마모와 미래 성능을 바꾸는 예를 사용했다. 특히 Mountain
Car에서 episode별 평균 속도에 비례해 다음 episode의 acceleration
force가 감소하는 실험도 수행했다.

따라서 **“행동으로 인한 미래 actuator 성능 변화” 자체는 새로운 개념이
아니다.**

- [Off-Policy Evaluation for Action-Dependent Non-stationary Environments,
  NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/3bf80b34f731313b8292f4578e820c90-Paper-Conference.pdf)

해당 연구가 주로 미래 정책 성능을 추정하는 off-policy evaluation을
다룬 반면, LifePhyBench는 embodied control, 숨은 건강 추론, 장기 보존
제어와 선택적 reset 평가를 다룬다.

### 5.2 변화하는 actuator dynamics

UBADA는 actuator dynamics가 지속적으로 또는 갑자기 바뀌는 환경을
Gymnasium wrapper로 제공한다. 따라서 “변화하는 actuator를 위한 범용
wrapper”도 단독 신규성으로 주장하기 어렵다.

- [Universal Benchmark for Actuation Dynamics Adaptation](https://openreview.net/forum?id=9RsXowObLi)

LifePhyBench에서는 actuator 변화가 단순히 주어진 조건으로 바뀌는 것이
아니라, 에이전트가 누적한 행동량에 의해 생성되는 건강 궤적이라는 점을
주된 실험축으로 삼는다.

### 5.3 손상 인지 로봇 벤치마크

가장 가까운 최신 연구는 RSS 2026의 OopsieVerse/DamageSim이다. 이
연구는 접촉력, 온도, 액체 노출로부터 mechanical, thermal, fluid
damage를 계산하고 32개 가정용 조작 작업에서 안전한 정책을 평가한다.

- [OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot
  Manipulation](https://arxiv.org/html/2606.31993)

이 연구와의 차이가 특히 중요하다. OopsieVerse 논문은 health를 관측,
보상, 종료 조건에 사용할 수 있도록 추가하면서도 기존 task dynamics는
그대로 유지한다고 명시한다. 반면 LifePhyBench에서는 누적 health가
actuator gain이나 damping을 변화시켜 다음 상태의 전이 자체를 바꾼다.

$$
\text{OopsieVerse: damage}\rightarrow
\{\text{observation, reward, termination}\}
$$

$$
\text{LifePhyBench: damage}\rightarrow
\text{future physical transition dynamics}
$$

이 차이는 현재 확인된 선행연구 중 가장 방어력이 높은 차별점이다.

### 5.4 Reset-free RL

CRONOS는 shared scene을 계속 유지하며 제한된 reset budget 아래에서
multi-task manipulation을 학습한다. Continuing SAC 연구는 episode
termination과 robot embodiment reset 없이 학습하는 문제를 다룬다.

- [CRONOS: Benchmarking Multi-Task Robotic Manipulation for Reset-Free
  RL](https://embodiedai-ntu.github.io/cronos/index.html)
- [Learning Without Time-Based Embodiment Resets in Soft-Actor
  Critic](https://proceedings.mlr.press/v330/farrahi26a.html)

그러나 이들은 주로 scene이나 embodiment를 아예 reset하지 않는 문제다.
LifePhyBench는 task state는 정상적으로 reset하면서 물리적 건강만
선택적으로 유지한다. 즉 “reset이 없음”보다 “서로 다른 상태 성분에
서로 다른 reset 규칙이 적용됨”을 연구한다.

### 5.5 생리적 피로와 물리적 변화

MyoSuite는 musculoskeletal control에서 muscle fatigue, sarcopenia 및
physiological variation을 지원한다. 따라서 “로봇 또는 생체 actuator의
피로를 시뮬레이션한다”는 주장 역시 새롭지 않다.

- [MyoSuite: A Contact-rich Simulation Suite for Musculoskeletal Motor
  Control](https://proceedings.mlr.press/v168/caggiano22a.html)

LifePhyBench의 차별화 목표는 특정한 muscle-fatigue 모델이 아니라,
다양한 물리 열화 메커니즘에 공통으로 적용되는 selective-reset 및
endogenous-dynamics 평가 프로토콜이다.

### 5.6 숨은 동역학 적응과 cross-episode memory

숨은 dynamics parameter를 추론하거나 recurrent state를 여러 episode에
걸쳐 유지하는 방법에도 많은 선행연구가 있다.

- [LILAC: Deep Reinforcement Learning amidst Continual Structured
  Non-Stationarity](https://proceedings.mlr.press/v139/xie21c.html)
- [CARL: A Benchmark for Contextual and Adaptive Reinforcement
  Learning](https://openreview.net/pdf?id=Y42xVBQusn)
- [Hidden-Parameter Markov Decision Processes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6814194/)
- [UP-OSI: Preparing for the Unknown](https://arxiv.org/abs/1702.02453)
- [RMA: Rapid Motor Adaptation for Legged Robots](https://www.roboticsproceedings.org/rss17/p011.html)
- [RL²: Fast Reinforcement Learning via Slow Reinforcement
  Learning](https://arxiv.org/abs/1611.02779)
- [SF-RSSM: Coordinating Fast and Slow Dynamics for Efficient World
  Modeling](https://ojs.aaai.org/index.php/AAAI/article/view/39825)

따라서 다음 요소들은 각각 단독으로 신규성을 갖지 않는다.

- 숨은 동역학을 추론하는 것
- RNN 기억을 episode 사이에 유지하는 것
- fast/slow memory를 사용하는 것
- 최근 상태·행동 이력으로 system identification을 수행하는 것

## 6. 차별성 유효성 판정

### 6.1 현재 방어 가능한 주장

가장 안전한 신규성 문구는 다음과 같다.

> 2026년 8월 현재 확인한 범위에서는, 행동·접촉으로 변화하는 숨은
> 물리적 건강 상태가 task reset 이후에도 선택적으로 유지되고, 그 건강
> 상태가 이후 transition dynamics를 변화시키며,
> persistence·endogeneity·observability를 독립적인 대조군으로 평가하는
> 공개 로봇학습 벤치마크는 확인하지 못했다.

이는 여러 요소의 **조합**에 대한 제한적인 주장이다. 각 개별 요소가
최초라는 의미는 아니며, 검색 결과만으로 관련 연구의 부재가 증명되는
것도 아니다.

### 6.2 주장별 평가

| 주장 | 판정 | 이유 |
|---|---|---|
| 행동이 마모를 만든다 | 신규성 없음 | NeurIPS 2022에서 직접 다룸 |
| 누적 damage를 측정한다 | 신규성 없음 | OopsieVerse, MyoSuite 등이 존재 |
| episode 사이에 memory를 유지한다 | 신규성 없음 | RL² 및 meta-RL 선행연구 존재 |
| task reset과 physical reset을 분리한다 | 조건부 유효 | 가장 가까운 reset-free 연구와 문제 구조가 다름 |
| 누적 health가 이후 물리 동역학을 바꾼다 | 강한 차별점 | OopsieVerse는 기존 task dynamics를 유지 |
| endogeneity와 persistence를 직교 대조한다 | 강한 실험 설계 차별점 | 효과의 원인을 인과적으로 분리할 수 있음 |
| action-aware health inference와 장기 보존을 분리 평가한다 | 조건부 유효 | 기존 system-ID와 구별되는 실험 필요 |
| fast/slow 신규 알고리즘 | 아직 무효 | 제안 알고리즘과 우위 결과가 아직 없음 |
| 실제 로봇의 물리적 노화를 재현한다 | 현재 주장 불가 | 현 모델은 phenomenological simulation |

종합하면, **벤치마크 및 평가 프로토콜의 차별성은 중간 이상이지만
알고리즘적 차별성은 아직 확립되지 않았다.**

## 7. 차별성을 입증하기 위한 필수 실험

논문에서 다음 대조군이 모두 포함돼야 차별점이 실험적으로 유효해진다.

1. Persistent health 대 episode-reset health
2. Endogenous action-driven degradation 대 dose-matched exogenous degradation
3. Dynamics-feedback damage 대 reward-only damage
4. Hidden health 대 privileged-health oracle
5. Observation-only 대 action-aware health estimator
6. Episode-reset RNN 대 lifetime-persistent RNN
7. 학습에서 보지 않은 degradation rate 대 보지 않은 함수형 법칙
8. Myopic health oracle 대 lifetime-planning oracle

특히 3번이 OopsieVerse와 구분되는 핵심이고, 1번과 2번이 기존
reset-free RL 및 일반적인 non-stationary RL과 구분되는 핵심이다.

## 8. 현재 연구 진행 상태

현재까지는 연구문제의 의미론과 구현 가능성을 확인하고, 초기 학습
기준선을 점검한 단계다.

- Pusher-v5와 Reacher-v5 지원
- wear, thermal derating, joint aging 구현
- episode/lifetime reset 의미 검증
- endogenous/exogenous 열화 대조군 구현
- PPO/SAC 및 recurrent PPO 개발 학습 수행
- 전체 단위 테스트 34개 통과

초기 recurrent 실험에서는 episode마다 memory를 초기화한 정책이 평균
`-31.30`, 20-task lifetime 동안 memory를 유지한 정책이 `-35.13`이었다.
개발 seed가 2개뿐이므로 과학적 결론은 아니지만, 단순히 memory를 오래
유지한다고 성능이 좋아지는 것은 아니라는 신호다.

이 결과는 다음 연구질문을 뒷받침한다.

> 어떤 조건에서 cross-episode memory가 유효하고, 언제 불필요하거나
> 최적화를 방해하는가?

현재 recurrent network의 weights는 배포 중 고정되어 있다. 따라서 이를
continual learning이라고 부르지 않고, frozen online adaptation 또는
latent-state inference로 구분한다.

## 9. 연구의 한계와 위험요인

현재 연구에는 다음과 같은 한계가 있다.

- 실제 장비의 재료 피로나 고장 데이터를 이용해 열화법칙을 보정하지 않았다.
- 현재 메커니즘은 물리적으로 영감을 받은 phenomenological model이다.
- 학습 결과가 주로 Pusher-v5와 actuator wear 조건에 집중돼 있다.
- 단순 recurrent baseline이 아직 lifetime memory의 이점을 보여주지 못했다.
- proposed method와 알고리즘적 우위가 아직 존재하지 않는다.
- 하나의 simulator나 scalar health 변수에만 결론이 의존하면 일반화 주장이 약하다.

따라서 논문에서는 다음 표현을 피해야 한다.

- 최초의 action-dependent non-stationary RL 문제
- 최초의 damage-aware, wear-aware 또는 fatigue-aware benchmark
- 최초의 cross-episode memory 또는 dual-timescale RL 방법
- 실제 로봇의 노화를 정확히 재현한다는 주장
- recurrent state만 변화하는 방법을 continual parameter learning으로 표현

물리적 보정 자료가 확보되기 전까지는 **physics-inspired degradation** 또는
**phenomenological degradation law**라는 표현을 사용해야 한다.

## 10. 최종 판단

연구를 중단해야 할 정도의 선행연구 중복은 발견되지 않았다. 다만
논문의 중심을 “로봇 마모를 처음 다룬다”에 두면 신규성 주장은 성립하지
않는다.

가장 방어력 있는 논문 중심축은 다음과 같다.

> 기존 연구가 damage 측정, 변화하는 dynamics, reset-free learning을
> 각각 다뤘다면, LifePhyBench는 행동으로 만들어진 숨은 물리적 건강
> 상태가 task reset을 가로질러 지속되고 이후 동역학을 변화시키는
> 상황을 인과적으로 분해하여 평가한다.

현재 차별성은 유효하지만 아직 입증 완료 단계는 아니다. 최종적으로는
여러 task와 열화 메커니즘에서 persistence와 endogeneity의 상호작용이
재현되고, reward-only damage나 일반 RNN/system-ID 기준선만으로 설명되지
않는 결과가 나와야 TMLR 수준의 기여가 된다.

## 11. 관련 프로젝트 문서

- [연구 명세](RESEARCH_SPEC.md)
- [신규성 및 선행연구 장부](NOVELTY_LEDGER.md)
- [전체 실험 계획](EXPERIMENT_PLAN.md)
- [Recurrent baseline 프로토콜](RECURRENT_BASELINE_PROTOCOL.md)
- [Recurrent 개발 결과](RECURRENT_DEVELOPMENT_RESULTS.md)

## 12. 2026-08-27 확정 실험 업데이트

위 8--10절은 초기 개발 단계의 기록이다. 이후 물리 task마다 Low/High를
한 번만 선택하는 계층형 thermal-commitment 과제를 설계하고, calibration
seed 5300--5304와 분리된 held-out 학습 seed 6300--6319에서 2 x 2 실험을
완료했다.

- 동적·내생적 thermal 조건에서 lifetime LSTM - task-reset LSTM:
  `+1.1269 reward/task`;
- seed bootstrap 95% CI: `[0.7950, 1.4328]`;
- held-out 학습 전에 로컬 manifest에 지정한 단측 t 검정:
  `p = 1.15e-6`;
- 정적 zero-dose 대조군의 memory 차이: 정확히 `0`.

단, task-reset 정책 20개가 모두 Always-Low로 수렴했다. 설계 단계에서
완전탐색한 task-reactive 규칙(첫 task만 High, 이후 Low)은 Always-Low보다
`+0.9407/task` 높다. 따라서 이번 결과의 정확한 해석은 “이 고정된 단일
Pusher 진단에서 lifetime-state RecurrentPPO가 동일 예산의 task-reset
RecurrentPPO보다 우수했다”이다. Cross-task memory가 필수이거나 정책이
숨은 thermal health를 추론했다는 결론은 아직 성립하지 않는다.

상세 프로토콜, 표, 한계는
[`HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md`](HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md)에
정리돼 있다.
