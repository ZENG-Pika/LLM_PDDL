(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5)
(:init
(arm-empty)
(on-table b5)
(on b3 b5)
(on b2 b3)
(on b1 b2)
(on b4 b1)
(clear b4)
)
(:goal
(and
(on b1 b3)
(on b3 b5))
)
)