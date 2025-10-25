#!/bin/bash

# Replace YOUR_GITHUB_USERNAME with your actual GitHub username
# For example: git remote add origin https://github.com/thomaskantecki/kinexis.git

echo "Enter your GitHub username:"
read username

git remote add origin "https://github.com/${username}/kinexis.git"
git branch -M main
git push -u origin main

echo "Repository pushed to GitHub successfully!"
echo "Your repository URL: https://github.com/${username}/kinexis"