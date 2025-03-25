# Navigate to the courses submodule
cd courses

# Copy the README.md content into a new file
# You can just create README.md and paste the content I provided

# Add and commit the README
git add README.md
git commit -m "Add README.md to courses submodule"

# Push the changes to the submodule repository
git push origin main  # or master, depending on your branch name

# Go back to the main repository directory
cd ..

# Update the reference to the submodule in the main repository
git add courses
git commit -m "Update courses submodule to include README.md"

# Push the changes to the main repository
git push origin main