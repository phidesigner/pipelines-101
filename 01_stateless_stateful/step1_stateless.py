# Step 1: a stateless rule: same input, same output, nothing stored
def minutes_to_hours(minutes):
    return minutes / 60


for minutes in [480, 900, 480]:
    print(minutes, "min ->", minutes_to_hours(minutes), "h")
