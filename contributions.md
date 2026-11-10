 I was responsible for all phases of design, implementation, evaluation, and reporting.

On the machine learning side, I implemented the data pipeline (make_data.py), the seeded Sudoku generator, and the CNN digit classifier alongside a logistic regression baseline (nn/), including temperature scaling for calibration.

For the logic components, I built the Constraint Satisfaction Problem solver (csp/) with switchable Backtracking, Forward Checking, and AC-3 (MAC), complete with backtrack and assignment counters. I also implemented the search-based correction module (search/), applying Uniform Cost Search, Greedy, and A* using a probability-drop cost function to resolve misread digits.

For integration and evaluation, I wrote the end-to-end pipeline and demo (run_demo.py), as well as the experiment scripts (eval/). I ran the three required experiments over five seeds, analyzed the results, and generated the tables and plots. Finally, I authored the complete report, data dictionary, and README, ensuring all system limitations and evaluation metrics were documented honestly.
