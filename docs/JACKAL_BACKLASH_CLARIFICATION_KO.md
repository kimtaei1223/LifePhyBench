# Clearpath Jackal의 누적 유격과 LifePhyBench의 관계

> 작성일: 2026-08-04  
> 목적: 실제 Jackal의 장기 열화와 시뮬레이션 연구의 범위를 혼동하지 않기

## 1. 동료 연구자에게 전달할 핵심 답변

그 이해는 **일부만 맞고 중요한 부분에서 다릅니다.** 현재 연구가 말하는
것은 렌더링된 로봇 어셋의 기어, 베어링, 윤활, 재료 피로를 유한요소해석
수준으로 장시간 연산하여, 실제 Jackal의 유격 발생량을 자동으로 계산하는
시뮬레이터는 아닙니다.

현재 구현은 다음과 같은 통제된 연구용 모델입니다.

1. 행동량 또는 접촉량으로부터 숨은 health state를 누적한다.
2. 그 health state를 actuator gain, thermal derating, joint damping 같은
   물리 파라미터에 연결한다.
3. 그 결과 동일한 command라도 이후의 물리 상태 전이가 달라지도록 한다.

즉, 핵심은 **현실의 특정 기어박스 마모량을 예측하는 digital twin**이 아니라,
**누적된 건강 상태가 미래 동역학에 피드백될 때 정책이 어떻게 달라져야
하는가**를 통제된 조건에서 연구하는 것이다.

실제 Jackal을 대상으로 하려면, 시뮬레이션 health와 실제 유격 사이의
매핑을 별도 실험으로 식별하고 보정해야 한다.

## 2. “몇 시간 운행하면 유격 차이가 생기나?”에 대한 정확한 답

현재 주어진 정보만으로는 답할 수 없다. 최대 속도 2.0 m/s와 일반적인
노면 마찰계수만으로 실제 유격 증가 시간을 계산하는 식은 없다.

Clearpath의 공개 Jackal 사양은 최대 속도 2.0 m/s, 190 mm wheel,
최대 payload 20 kg, 기본 사용 8시간 및 고부하 사용 2시간의 운용시간을
제시한다. 또한 wheel odometry, motor current, IMU, GPS를 내부 sensing으로
제시한다. 그러나 이 공개 사양만으로는 특정 Jackal의 gearhead 구조,
초기 backlash, 윤활 상태, 허용 유격, 하중별 마모 곡선 또는 교체 수명을
알 수 없다.

- [Clearpath Jackal User Manual](https://docs.clearpathrobotics.com/docs_robots/legacy/ros1_robots/outdoor_robots/jackal/user_manual_jackal/)

속도만으로 계산 가능한 것은 이동거리뿐이다.

| 연속 주행 시간 | 2.0 m/s에서의 이상적 이동거리 |
|---:|---:|
| 1시간 | 7.2 km |
| 2시간 | 14.4 km |
| 8시간 | 57.6 km |

위 거리는 속도가 항상 2.0 m/s라고 가정한 값이다. 제조사의 2시간 고부하
운용시간은 battery/runtime 사양이지, 2시간 동안 최고 속도를 지속할 수
있다는 보증이나 유격 수명 사양은 아니다.

따라서 다음과 같은 단정은 하면 안 된다.

> “일반 마찰 노면에서 최고속도 3시간 운행하면 Jackal 유격이 몇 mm
> 증가한다.”

이 값은 공개 사양으로부터 도출할 수 없고, 실제 장비별 측정 없이는
책임 있게 제시할 수 없다.

## 3. 왜 속도와 마찰만으로 계산할 수 없는가

기어 유격 또는 drivetrain backlash의 증가량은 단순 주행시간보다 다음
변수에 훨씬 민감하다.

- 실제 gearhead 구조, gear material, 초기 조립 공차 및 preload
- 모터축과 wheel encoder의 위치
- 윤활 상태, 온도, 먼지·수분 유입
- payload와 무게중심
- 경사, 충격, 요철, 연석 충돌 등 peak load
- 가속·감속 횟수와 전후진 방향전환 횟수
- skid-steer 회전으로 발생하는 scrub torque
- 제어기의 torque/current saturation 및 보호 로직

같은 10 km라도 일정 속도 직진 주행과 반복적인 급정지·후진·제자리 회전은
gear tooth contact와 torque reversal의 횟수 및 peak load가 크게 다르다.
따라서 누적 사용량은 단순 시간보다 최소한 다음과 같은 dose로 모델링하는
것이 더 타당하다.

$$
z_{t+1}=z_t+f(|\tau_t|, |\Delta\omega_t|,
\mathrm{reversal}_t, \mathrm{shock}_t, T_t, \mathrm{terrain}_t)
$$

여기서 $z_t$는 실제 유격 그 자체가 아니라, 마모·열·충격을 요약한
추정 health state가 될 수 있다.

## 4. 일정 속도 직진과 유격의 관계

기어 유격은 특히 torque 방향이 바뀔 때 관찰되기 쉽다. 모터축의 회전
방향이 바뀌면 gear tooth 사이의 빈 공간을 먼저 메워야 wheel에 torque가
전달되기 때문이다.

따라서 일정한 방향으로 2.0 m/s 직진하는 경우에는 다음이 일반적이다.

- 기존의 유격은 출발 직후 한 번 흡수될 수 있다.
- 정상상태 속도에서는 velocity controller가 encoder feedback으로 일부
  오차를 보상할 수 있다.
- 유격 자체보다 타이어 slip, 타이어 반경 변화, 노면, battery voltage,
  온도, 제어기 tuning이 외부 궤적오차에 더 크게 보일 수 있다.

반면 다음 조건에서는 유격이나 transmission compliance의 영향이 더 잘
드러난다.

- 정·역방향 반복 전환
- 작은 속도 명령의 부호 전환
- stop-and-go 주행
- 제자리 회전과 잦은 곡선 주행
- 큰 payload, 경사, 거친 지면 또는 충격

따라서 “몇 시간 뒤 최고속도 직진이 느려지는가?”보다 “몇 회의 torque
reversal 및 어떤 peak load 이후 reversal deadband 또는 transient tracking
error가 유의하게 증가하는가?”가 유격에는 더 적절한 질문이다.

## 5. LifePhyBench와 실제 Jackal 모델의 관계

### 5.1 현재 LifePhyBench가 하는 일

현재 연구에서는 예를 들어 다음과 같은 현상론적 관계를 정의한다.

$$
\tau_t^{\mathrm{actual}}=g(z_t)\tau_t^{\mathrm{command}}
$$

$$
b_t=b_0+\Delta b(z_t)
$$

첫 식은 actuator health가 나빠질수록 실제 torque가 감소하는 모델이고,
둘째 식은 health에 따라 damping이 증가하는 모델이다. 이 모델은 동일한
state와 command에서 다음 물리 상태가 달라지게 만든다.

$$
P(x_{t+1}\mid x_t,a_t,z_{\mathrm{healthy}})
\neq
P(x_{t+1}\mid x_t,a_t,z_{\mathrm{degraded}})
$$

이것은 연구 문제를 명확히 하는 데는 유효하지만, 아직 Jackal의 실제
gearbox에서 몇 km 주행 후 얼마의 backlash가 생기는지를 예측하는 모델은
아니다.

### 5.2 실제 Jackal에 연결하려면 필요한 추가 단계

실제 장비로 논리를 연결하려면 다음 순서가 필요하다.

1. **현상 정의:** 무엇을 열화 결과로 볼지 정한다. 예: reversal deadband,
   wheel-speed step-response delay, motor-current asymmetry, external
   trajectory error.
2. **기준선 측정:** 새롭거나 정비 직후 상태에서 동일한 주행 시험을 여러
   번 수행해 불확실성을 측정한다.
3. **사용량 기록:** 거리만이 아니라 torque/current, 속도 변화,
   방향전환, 온도, payload, terrain 및 충격 사건을 기록한다.
4. **반복 측정:** 충분한 사용량 block 뒤에 동일 시험을 반복한다.
5. **식별:** health proxy와 실제 측정량의 관계 및 신뢰구간을 추정한다.
6. **시뮬레이터 보정:** 추정된 범위만 simulator의 degradation law에
   반영하고, 보정 범위 밖으로 일반화하지 않는다.

## 6. 권장 측정 프로토콜

의도적으로 장비를 마모시키기 위해 최고속도로 수 시간 반복 주행하는 것은
권장하지 않는다. 안전·장비 관리 목적의 정상 운용 로그와 짧은 진단 시험을
우선 사용해야 한다. Clearpath도 고속 운용 시 넓고 장애물이 없는 공간을
권장하며, 시스템 상태는 `/status`와 `/diagnostics`로 관찰하도록 안내한다.

- [Clearpath Jackal Driving Guide](https://clearpathrobotics.com/assets/guides/foxy/jackal/JackalDriving.html)
- [Clearpath Jackal User Manual: performance considerations](https://docs.clearpathrobotics.com/docs_robots/legacy/ros1_robots/outdoor_robots/jackal/user_manual_jackal/)

진단 실험의 최소 구성은 다음과 같다.

| 항목 | 권장 방식 |
|---|---|
| 직진 시험 | 동일 노면·payload에서 속도 step을 반복 |
| 유격 민감 시험 | 작은 정·역 속도 또는 torque command를 반복 |
| 회전 시험 | 좌·우 회전 및 제자리 회전의 yaw response 비교 |
| 기준 센서 | wheel odometry, IMU, motor current, battery voltage, temperature |
| 외부 기준 | 가능하면 Vicon, AprilTag, RTK-GNSS 또는 고정 카메라 추적 |
| 주요 지표 | reversal delay, deadband, rise time, overshoot, yaw error, 외부 궤적오차 |

Jackal은 wheel odometry, motor current, IMU, GPS를 제공하지만, 유격을
wheel encoder만으로 확정할 수 있는지는 encoder가 drivetrain의 어느 쪽에
있는지에 따라 달라진다. 따라서 외부 위치 기준을 함께 두는 것이 중요하다.

Clearpath의 Gazebo 모델은 wheel slip, skidding, inertia의 합리적 근사를
포함한다고 안내한다. 이는 일반 주행 dynamics 검증에는 유용하지만, 실제
개별 장비의 장기 gear backlash 식별을 대신하지는 않는다.

- [Clearpath Jackal simulation documentation](https://docs.clearpathrobotics.com/docs_robots/legacy/ros1_robots/outdoor_robots/jackal/tutorials_jackal/)

## 7. 결론

현 시점에서의 정확한 답은 다음과 같다.

> 일반 마찰 노면에서 Jackal을 2.0 m/s로 몇 시간 주행하면 유격 때문에
> 실제 주행 차이가 생기는지는, 공개 사양과 속도만으로 계산할 수 없다.
> 보통의 직진 주행 수 시간 안에 유격 변화가 명확히 관찰될 것이라고
> 가정해서도 안 된다. 유격은 특히 방향전환과 충격·고하중 조건에서
> 민감하게 나타날 수 있으며, 실제 임계 사용량은 해당 장비의 drivetrain
> 상태와 운용 이력의 실측으로 정해야 한다.

따라서 LifePhyBench는 현재 **실제 Jackal의 수명 예측 모델**이 아니라,
**행동으로 변화하는 숨은 health가 이후 동역학을 바꿀 때의 학습 및
계획 문제를 통제하는 벤치마크**로 위치시키는 것이 정확하다. 실제 Jackal
실험은 이 벤치마크의 물리 파라미터 범위를 보정하거나 대표적인 현실성
검사를 수행하는 후속 단계가 된다.

