import pandas as pd
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

# Loading of dataset
df = pd.read_csv("Dataset - Admission_Predict.csv")

# Plot distributions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Interest'], bins=20)
plt.title('Interest Distribution')

plt.subplot(1, 2, 2)
plt.hist(df['CGPA'], bins=20)
plt.title('CGPA Distribution')
plt.show()

# Preparation of data
students = [f"s{id}" for id in df['Student ID']]
courses = [f"c{uni}" for uni in df['University Course'].unique()]

# Creation of dictionaries for interest and score
interest = {}
score = {}
for _, row in df.iterrows():
    student_id = f"s{row['Student ID']}"
    course_id = f"c{row['University Course']}"

    interest[(student_id, course_id)] = row['Interest']
    score[(student_id, course_id)] = row['CGPA']

# Initialization of the model
model = Model("EnrollmentOptimization")

# Creation of the decision variables only for eligible student-course pairs
x = {}
for (i, j), interest_val in interest.items():
    if interest_val >= 0.8 and score[(i, j)] >= 7:
        x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# Course capacities constraint
capacity = {course: 20 for course in courses}

# Addition of capacity + eligibility constraints per course
for j in courses:
    model.addConstr(
        quicksum(x[(i, j)] for (i, j_) in x if j_ == j and interest[(i, j_)] >= 0.8 and score[(i, j_)] >= 7)
        <= capacity[j],
        name=f"capacity_and_eligibility_{j}"
    )


# Objective function (weighted sum of interest and score)
alpha = 0.5  # weight for interest
beta = 0.5   # weight for score

model.setObjective(
    quicksum((alpha * interest[i, j] + beta * score[i, j]) * x[(i, j)] for (i, j) in x),
    GRB.MAXIMIZE
)

# Optimization
model.optimize()

# Output of results
if model.status == GRB.OPTIMAL:
    print("\nOptimal Enrollments:")
    total_interest = 0
    total_score = 0
    enrolled_count = 0

    for (i, j) in x:
        if x[(i, j)].x > 0.5:
            print(f"Student {i} → Course {j} | Interest: {interest[(i, j)]:.2f}, Score: {score[(i, j)]:.2f}")
            total_interest += interest[(i, j)]
            total_score += score[(i, j)]
            enrolled_count += 1

    print(f"\nTotal Enrolled: {enrolled_count}")
    print(f"Total Interest: {total_interest:.2f}")
    print(f"Total Score: {total_score:.2f}")
    print(f"Combined Objective: {model.objVal:.2f}")
else:
    print("No solution found.")
