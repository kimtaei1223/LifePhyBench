# v12.3 factorial 및 보상 민감도 — 논문 삽입용 요약

독립적인 fresh lifetime seed 100개를 사용한 2 x 2 분석에서, 동일한 불확실성 마진 z=1.5에서 residual의 효과는 **+0.103 reward/task**였다(95% bootstrap CI [-0.004, 0.215], paired sign-flip p=0.0681). 반면 residual이 없는 경우 불확실성 마진의 효과는 **+0.965**(95% CI [0.733, 1.196]), residual이 있는 경우에는 **+0.736**(95% CI [0.524, 0.950])였다. 상호작용은 -0.229 (95% CI [-0.407, -0.051])로 음수였다. 따라서 전체 이득을 residual의 독립 효과로 귀속하지 않으며, 주된 확증 결론은 불확실성 마진을 포함한 belief supervision의 효과이다.

원래의 hybrid z=1.5 대 physics z=0 차이 +1.068은 base task return -1.191, throughput bonus -0.029, avoided-trip contribution +2.288으로 분해되었다. 즉, 평균 효용 이득은 주로 thermal trip 감소에서 발생하며 즉시 task 성능의 희생을 포함한다.

동일 궤적을 사후 재가중한 민감도 분석에서 원래 throughput bonus 2를 유지할 때 평균 break-even trip penalty는 약 **40.0**였다. penalty 25에서는 총 효과가 -0.457, 50에서는 +0.306, 75에서는 +1.068, 100에서는 +1.831이었다. 따라서 효용 개선은 thermal trip에 중간 이상의 비용을 부여하는 응용에서 성립하며, trip을 거의 중요하지 않게 취급하는 목적함수에는 일반화되지 않는다. 이 분석은 정책을 다른 reward로 재학습한 결과가 아니라 고정된 평가 궤적의 회계적 민감도 분석이다.
