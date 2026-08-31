# 누적 Damage 측정과 Physical-Dynamics Feedback의 차이

## 1. 핵심 요약

두 개념의 핵심 차이는 **누적값을 기록만 하는가**, 아니면 **그 누적값이
다음 물리 상태를 실제로 변화시키는가**이다.

- **Damage measurement:** damage를 누적하여 관측, 보상, 종료 조건 또는
  평가 지표로 사용하지만 원래 물리 동역학은 그대로 유지한다.
- **Dynamics feedback:** 누적 health가 actuator gain, joint damping 등의
  물리 파라미터를 변화시켜 이후 상태 전이에 직접 영향을 준다.

따라서 누적 여부 자체가 차이가 아니라, 누적 변수가 transition kernel에
들어가는지가 차이이다.

## 2. 누적 Damage 측정

누적 damage 측정에서는 로봇의 행동이나 접촉으로 발생한 손상을 별도의
변수로 계속 기록한다.

$$
d_{t+1}=d_t+\Delta d(x_t,a_t)
$$

예를 들어 접촉력이 임계값을 넘을 때마다 damage가 증가할 수 있다.

$$
\Delta d_t=\max(F_t-F_{\mathrm{threshold}},0)
$$

이 damage는 다음과 같이 사용할 수 있다.

- observation에 health bar로 표시
- reward에서 penalty로 차감
- damage가 임계값을 넘으면 episode 종료
- 성공했지만 안전하지 않은 동작을 판별하는 평가 지표

그러나 로봇의 원래 물리 동역학은 그대로일 수 있다.

$$
x_{t+1}\sim P(x_{t+1}\mid x_t,a_t)
$$

즉 damage가 0이든 80이든, 같은 상태에서 같은 행동을 하면 로봇은 같은
힘과 속도로 움직인다.

예를 들어 접시를 반복해서 세게 내려놓으면 damage 점수는 증가하지만,
접시의 질량·마찰·강성이나 로봇의 actuator 출력은 변하지 않을 수 있다.
손상은 측정되지만 이후 물리적 행동 능력에는 반영되지 않는 것이다.

이를 간단히 표현하면 다음과 같다.

$$
\text{행동}\rightarrow\text{damage 기록}
\rightarrow\{\text{보상, 관측, 종료}\}
$$

## 3. 누적 Health가 물리 Dynamics를 변경하는 경우

LifePhyBench가 목표로 하는 설정에서는 누적 health가 단순한 평가
지표가 아니라 물리 시스템 상태의 일부이다.

$$
h_{t+1}=f(h_t,x_t,a_t)
$$

그리고 이 health가 다음 task state의 전이에 직접 들어간다.

$$
x_{t+1}\sim P(x_{t+1}\mid x_t,a_t,h_t)
$$

따라서 동일한 위치와 속도에서 동일한 행동을 수행해도 health가 다르면
다음 상태가 달라진다.

$$
P(x_{t+1}\mid x_t,a_t,h_{\mathrm{healthy}})
\neq
P(x_{t+1}\mid x_t,a_t,h_{\mathrm{degraded}})
$$

예를 들어 nominal actuator torque를 다음과 같이 정의할 수 있다.

$$
\tau_t^{\mathrm{nominal}}=K a_t
$$

마모가 누적되면 실제 torque는 다음처럼 감소할 수 있다.

$$
\tau_t^{\mathrm{actual}}=(1-d_t)K a_t
$$

| 누적 damage | 같은 action | 실제 torque |
|---:|---:|---:|
| 0.0 | 1.0 | 100% |
| 0.3 | 1.0 | 70% |
| 0.7 | 1.0 | 30% |

같은 action을 출력해도 마모된 로봇은 물체를 충분히 밀지 못하거나 목표
지점에 늦게 도달한다. 누적 damage가 미래의 transition dynamics를 바꾼
것이다.

$$
\text{행동}\rightarrow\text{health 변화}
\rightarrow\text{물리 파라미터 변화}
\rightarrow\text{미래 운동 변화}
$$

## 4. 수학적 차이

### 4.1 Damage 측정형

Health는 누적되지만 task transition에는 들어가지 않는다.

$$
h_{t+1}=f(h_t,x_t,a_t)
$$

$$
x_{t+1}\sim P(x_{t+1}\mid x_t,a_t)
$$

### 4.2 Dynamics-feedback형

Health가 task transition을 직접 조건화한다.

$$
h_{t+1}=f(h_t,x_t,a_t)
$$

$$
x_{t+1}\sim P(x_{t+1}\mid x_t,a_t,h_t)
$$

## 5. 두 설정의 비교

| 구분 | Damage 측정형 | Dynamics-feedback형 |
|---|---|---|
| damage 누적 | 있음 | 있음 |
| reward penalty | 가능 | 가능 |
| 관측에 health 제공 | 가능 | 가능 |
| damage에 따른 종료 | 가능 | 가능 |
| actuator 출력 변화 | 없음 | 있음 |
| damping·friction 등 변화 | 없음 | 있음 |
| 같은 상태·행동에서 health별 다음 상태 차이 | 없음 | 있음 |
| 장기 system identification 필요성 | 상대적으로 낮음 | 높음 |
| 현재 행동과 미래 제어능력의 trade-off | 보상 설계에 의존 | 물리적으로 발생 |

종료 조건도 미래 trajectory를 중단한다는 의미에서는 결과에 영향을
주지만, 작동 가능한 상태에서 health가 힘, 가속도, damping 또는 접촉
반응을 바꾸는 것과는 구분해야 한다. Dynamics feedback의 판정 기준은
종료 이전에도 health에 따라 물리 상태 전이가 달라지는지 여부이다.

## 6. LifePhyBench의 구체적인 예

### 6.1 Actuator wear

누적 wear가 actuator gain을 감소시킨다.

$$
\mathrm{gain}_t=\mathrm{gain}_0\cdot g(w_t)
$$

따라서 같은 action에서도 실제 힘이 감소한다.

### 6.2 Thermal derating

행동 부하로 온도가 상승하면 actuator gain이 일시적으로 감소한다.
Episode 경계에서 일부 냉각되고, lifetime reset에서는 완전히
초기화된다.

$$
\mathrm{gain}_t=\mathrm{gain}_0\cdot g(T_t)
$$

이는 영구적인 wear와 다른 시간 척도를 갖는 회복 가능한 열화이다.

### 6.3 Joint aging

누적 aging이 관절 damping을 증가시킨다.

$$
b_t=b_0+\Delta b(h_t)
$$

같은 torque를 적용해도 관절 움직임이 더 느려지고 에너지 손실이
커진다.

## 7. Episode를 넘어설 때 생기는 차이

한 episode 안에서만 damage penalty를 주고 reset 때 damage를 없애면
에이전트는 매번 새 장비를 받는 것과 유사하다.

LifePhyBench에서는 task 위치만 초기화되고 health는 남는다.

```text
Episode 1:
큰 action → 빠른 성공 → wear 축적

Episode reset:
물체와 로봇 자세 초기화
wear는 유지

Episode 2:
같은 action → actuator 출력 감소 → 성능 저하
```

따라서 Episode 1의 행동이 Episode 2의 물리적 난이도를 바꾼다. 이것이
cross-episode physical consequence이다.

## 8. 가장 직접적인 검증 방법

두 설정의 차이는 다음 counterfactual test로 검증할 수 있다.

1. 두 환경의 task state $x$를 동일하게 맞춘다.
2. 두 환경의 health만 다르게 설정한다.
3. 동일한 action $a$를 입력한다.
4. 다음 상태 $x'$를 비교한다.

$$
x_t^{(A)}=x_t^{(B)},\qquad
a_t^{(A)}=a_t^{(B)}
$$

$$
h_t^{(A)}\neq h_t^{(B)}
$$

이때 다음 조건이 성립해야 dynamics feedback이 존재한다.

$$
x_{t+1}^{(A)}\neq x_{t+1}^{(B)}
$$

확률적인 환경에서는 단일 next state 대신 반복 표본에서 추정한 조건부
분포를 비교해야 한다.

$$
P(x_{t+1}\mid x_t,a_t,h_t^{(A)})
\neq
P(x_{t+1}\mid x_t,a_t,h_t^{(B)})
$$

반대로 health bar와 reward만 달라지고 다음 위치, 속도, 접촉 결과가
같다면 이는 damage 측정형이지 dynamics-feedback형은 아니다.

## 9. 연구적으로 중요한 이유

Damage 측정형에서 정책은 대체로 다음 문제를 푼다.

> 현재 작업 보상과 현재 damage penalty 사이에서 어떤 행동을 선택할
> 것인가?

Dynamics-feedback형에서는 문제가 더 장기적으로 바뀐다.

> 현재 작업을 빠르게 끝내기 위해 장비를 소모할 것인가, 아니면 미래
> 작업 수행 능력을 보존할 것인가?

후자에서 health가 숨겨져 있다면 에이전트는 행동 결과를 통해 health를
추론해야 하고, 현재 행동이 미래의 관측 가능성과 제어 능력까지 바꾼다.
이에 따라 다음 요소가 중요해진다.

- action-aware health inference
- cross-episode memory
- 장기 credit assignment
- degradation-aware planning
- 현재 성과와 미래 가용성의 trade-off

## 10. 필요한 대조 실험

Dynamics feedback의 기여를 보이려면 다음 조건을 같은 damage trajectory와
가능한 한 같은 reward scale에서 비교해야 한다.

1. **Measurement-only:** health를 기록하고 reward/observation에만 사용
2. **Dynamics-feedback:** 동일한 health가 물리 파라미터까지 변경
3. **No-damage control:** health 누적과 관련 penalty를 모두 제거

Measurement-only와 dynamics-feedback의 차이가 장기 정책, health 추정,
미래 성능에서 나타나야 물리 동역학 피드백의 추가 난이도를 주장할 수
있다.

## 11. 표현상의 주의사항

LifePhyBench의 gain 및 damping 변화는 아직 실제 특정 로봇의 열화
데이터로 보정된 모델이 아니다. 따라서 다음과 같이 표현하는 것이
정확하다.

> 누적 건강이 미래 동역학에 피드백되는 학습 문제를 통제된 방식으로
> 구현한다.

다음과 같은 표현은 실제 장비를 이용한 보정 증거가 확보되기 전까지
피해야 한다.

> 실제 로봇의 마모 또는 재료 손상을 정확하게 재현한다.

현재 열화법칙은 **physics-inspired** 또는 **phenomenological degradation
law**로 기술하는 것이 적절하다.

