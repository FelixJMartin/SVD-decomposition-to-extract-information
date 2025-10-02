### Task 2 — Results
- For digits 3 and 8, the singular values decay rapidly → low effective rank.
- Leading singular images capture the prototype digit; subsequent modes capture systematic handwriting variations.

| Singular values $A_3$ | Singular values $A_8$ |
| --- | --- |
| ![Singular values of $A_3$](assets/img/singular_vals_A3.png) | ![Singular values of $A_8$](assets/img/singular_vals_A8.png) |

**Digit 3 — first three singular images**

| $u_1$ | $u_2$ | $u_3$ |
| --- | --- | --- |
| ![3 $u_1$](assets/img/digit_3_image_1.png) | ![3 $u_2$](assets/img/digit_3_image_2.png) | ![3 $u_3$](assets/img/digit_3_image_3.png) |

**Digit 8 — first three singular images**

| $u_1$ | $u_2$ | $u_3$ |
| --- | --- | --- |
| ![8 $u_1$](assets/img/digit_8_image_1.png) | ![8 $u_2$](assets/img/digit_8_image_2.png) | ![8 $u_3$](assets/img/digit_8_image_3.png) |

---

### Task 3 — Results
- One SVD basis per digit using the top $k$ left singular vectors ($k=5\ldots 15$).
- Classification by smallest projection residual; per-digit accuracy increases with $k$ and then plateaus.

![Per-digit accuracy vs $k$](assets/img/precentages_of_success.png)
