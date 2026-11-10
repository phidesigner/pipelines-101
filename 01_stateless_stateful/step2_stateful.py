# Step 2: a stateful step: the fill value is learned from the data
import statistics  # new

hours = [2.5, 15.0, 8.0, 22.5, None, 4.0, 30.0, 12.0]  # new: customer 5 has no hours

known = [h for h in hours if h is not None]  # new
median = statistics.median(known)  # new: learned from these rows
print("learned median:", median)

filled = [median if h is None else h for h in hours]  # new: fill the gap
print("filled:", filled)
