# MedSafe-Dx Case Breakdown Analysis

This report analyzes model performance across different case types.

## Overall Performance by Model

| Model | N | Safety Pass | Missed Esc | Overconf Wrong | Unsafe Reassure | Top-1 | Top-3 |
|-------|---|-------------|------------|----------------|-----------------|-------|-------|
| anthropic-claude-haiku-4.5 | 250 | 95.6% | 4.4% | 0.0% | 0.0% | 39.2% | 58.4% |
| anthropic-claude-sonnet-4.5 | 250 | 87.2% | 7.6% | 2.8% | 3.2% | 58.4% | 77.2% |
| deepseek-deepseek-chat-v3-0324 | 250 | 85.2% | 7.2% | 4.0% | 4.0% | 40.8% | 60.0% |
| google-gemini-2.0-flash | 250 | 89.6% | 10.4% | 0.0% | 0.0% | 34.4% | 53.6% |
| google-gemini-3-pro-preview | 250 | 72.0% | 20.0% | 4.0% | 4.0% | 49.2% | 60.8% |
| openai-gpt-4.1 | 250 | 87.6% | 5.6% | 4.8% | 2.0% | 53.2% | 73.2% |
| openai-gpt-4o-mini | 250 | 92.4% | 6.0% | 0.4% | 1.2% | 28.8% | 47.6% |
| openai-gpt-5-chat | 250 | 94.0% | 3.2% | 2.4% | 0.4% | 53.2% | 73.2% |
| openai-gpt-5-mini | 250 | 90.0% | 10.0% | 0.0% | 0.0% | 55.2% | 68.4% |
| openai-gpt-5.2 | 250 | 97.6% | 2.0% | 0.4% | 0.0% | 52.4% | 68.4% |
| openai-gpt-oss-120b | 250 | 85.2% | 7.2% | 6.4% | 1.6% | 47.2% | 67.2% |

## Performance by Escalation Requirement (All Models)

| Stratum | N | Safety Pass | Missed Esc | Top-1 | Top-3 |
|---------|---|-------------|------------|-------|-------|
| Requires Escalation | 1716 | 85.4% | 12.2% | 46.0% | 62.4% |
| No Escalation | 1034 | 94.3% | 0.0% | 47.5% | 67.6% |

## Performance by Case Severity (All Models)

Severity categories:
- **Critical**: Severity 1-2 (life-threatening conditions)
- **Moderate**: Severity 3 (serious conditions)
- **Mild**: Severity 4-5 (less severe conditions)

| Severity | N | Safety Pass | Missed Esc | Overconf Wrong | Top-1 | Top-3 |
|----------|---|-------------|------------|----------------|-------|-------|
| Critical | 880 | 94.0% | 4.0% | 2.0% | 53.2% | 70.7% |
| Moderate | 836 | 89.7% | 8.7% | 1.4% | 48.3% | 65.1% |
| Mild | 990 | 83.0% | 10.2% | 3.2% | 39.7% | 57.6% |

## Performance by Symptom Count (All Models)

| Symptom Count | N | Safety Pass | Overconf Wrong | Top-1 | Top-3 |
|---------------|---|-------------|----------------|-------|-------|
| Low | 935 | 90.5% | 1.2% | 54.4% | 73.5% |
| Medium | 990 | 84.3% | 3.7% | 34.0% | 50.1% |
| High | 825 | 92.1% | 1.8% | 52.6% | 71.2% |

## Performance on Escalation-Required Cases (by Model)

| Model | N | Safety Pass | Missed Esc | Top-1 | Top-3 |
|-------|---|-------------|------------|-------|-------|
| anthropic-claude-haiku-4.5 | 156 | 92.9% | 7.1% | 38.5% | 60.3% |
| anthropic-claude-sonnet-4.5 | 156 | 85.3% | 12.2% | 59.6% | 75.0% |
| deepseek-deepseek-chat-v3-0324 | 156 | 84.6% | 11.5% | 34.0% | 50.6% |
| google-gemini-2.0-flash | 156 | 83.3% | 16.7% | 29.5% | 48.1% |
| google-gemini-3-pro-preview | 156 | 64.1% | 32.1% | 49.4% | 59.6% |
| openai-gpt-4.1 | 156 | 85.9% | 9.0% | 48.1% | 67.9% |
| openai-gpt-4o-mini | 156 | 89.7% | 9.6% | 24.4% | 41.7% |
| openai-gpt-5-chat | 156 | 92.3% | 5.1% | 51.3% | 69.2% |
| openai-gpt-5-mini | 156 | 84.0% | 16.0% | 57.7% | 70.5% |
| openai-gpt-5.2 | 156 | 96.2% | 3.2% | 59.6% | 70.5% |
| openai-gpt-oss-120b | 156 | 81.4% | 11.5% | 53.8% | 73.1% |

## Performance on Critical Severity Cases (by Model)

| Model | N | Safety Pass | Missed Esc | Top-1 | Top-3 |
|-------|---|-------------|------------|-------|-------|
| anthropic-claude-haiku-4.5 | 80 | 100.0% | 0.0% | 35.0% | 66.2% |
| anthropic-claude-sonnet-4.5 | 80 | 93.8% | 2.5% | 67.5% | 85.0% |
| deepseek-deepseek-chat-v3-0324 | 80 | 95.0% | 5.0% | 37.5% | 58.8% |
| google-gemini-2.0-flash | 80 | 92.5% | 7.5% | 32.5% | 56.2% |
| google-gemini-3-pro-preview | 80 | 85.0% | 15.0% | 65.0% | 78.8% |
| openai-gpt-4.1 | 80 | 92.5% | 1.2% | 53.8% | 72.5% |
| openai-gpt-4o-mini | 80 | 96.2% | 2.5% | 23.8% | 41.2% |
| openai-gpt-5-chat | 80 | 97.5% | 0.0% | 65.0% | 76.2% |
| openai-gpt-5-mini | 80 | 92.5% | 7.5% | 70.0% | 80.0% |
| openai-gpt-5.2 | 80 | 98.8% | 0.0% | 71.2% | 82.5% |
| openai-gpt-oss-120b | 80 | 90.0% | 2.5% | 63.7% | 80.0% |

## Statistical Tests

Chi-square tests for significant differences in safety pass rate:

- **anthropic-claude-haiku-4.5** - Escalation vs No-Escalation: χ²=5.36, p=0.0206 (**significant**)
- **anthropic-claude-haiku-4.5** - Critical vs Mild Severity: χ²=3.74, p=0.0530 (not significant)
- **anthropic-claude-sonnet-4.5** - Escalation vs No-Escalation: χ²=0.98, p=0.3224 (not significant)
- **anthropic-claude-sonnet-4.5** - Critical vs Mild Severity: χ²=6.54, p=0.0106 (**significant**)
- **deepseek-deepseek-chat-v3-0324** - Escalation vs No-Escalation: χ²=0.02, p=0.8796 (not significant)
- **deepseek-deepseek-chat-v3-0324** - Critical vs Mild Severity: χ²=13.96, p=0.0002 (**significant**)
- **google-gemini-2.0-flash** - Escalation vs No-Escalation: χ²=15.74, p=0.0001 (**significant**)
- **google-gemini-2.0-flash** - Critical vs Mild Severity: χ²=0.97, p=0.3251 (not significant)
- **google-gemini-3-pro-preview** - Escalation vs No-Escalation: χ²=11.81, p=0.0006 (**significant**)
- **google-gemini-3-pro-preview** - Critical vs Mild Severity: χ²=13.83, p=0.0002 (**significant**)
- **openai-gpt-4.1** - Escalation vs No-Escalation: χ²=0.73, p=0.3930 (not significant)
- **openai-gpt-4.1** - Critical vs Mild Severity: χ²=6.00, p=0.0143 (**significant**)
- **openai-gpt-4o-mini** - Escalation vs No-Escalation: χ²=3.22, p=0.0726 (not significant)
- **openai-gpt-4o-mini** - Critical vs Mild Severity: χ²=1.66, p=0.1977 (not significant)
- **openai-gpt-5-chat** - Escalation vs No-Escalation: χ²=1.38, p=0.2394 (not significant)
- **openai-gpt-5-chat** - Critical vs Mild Severity: χ²=2.08, p=0.1497 (not significant)
- **openai-gpt-5-mini** - Escalation vs No-Escalation: χ²=15.00, p=0.0001 (**significant**)
- **openai-gpt-5-mini** - Critical vs Mild Severity: χ²=0.09, p=0.7621 (not significant)
- **openai-gpt-5.2** - Escalation vs No-Escalation: χ²=2.24, p=0.1341 (not significant)
- **openai-gpt-5.2** - Critical vs Mild Severity: χ²=0.00, p=1.0000 (not significant)
- **openai-gpt-oss-120b** - Escalation vs No-Escalation: χ²=3.96, p=0.0466 (**significant**)
- **openai-gpt-oss-120b** - Critical vs Mild Severity: χ²=3.75, p=0.0527 (not significant)
