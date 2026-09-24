# Classroom matching demo

The public notebook and browser demo use **entirely fictional profiles** from `student_data_sample.json`. The traits illustrate different similarity measures; they are not a record of students or evidence that a recommended team will work well. `sample-responses.csv` illustrates the survey format with fictional entries.

Run the Lecture 2 notebook from the repository, or serve this directory with a local HTTP server and open `index.html`. An HTTP server is needed for the browser to load the sample JSON.

Keep real survey responses, contact information, and class rosters local. The paths `response.csv`, `student_data.json`, and dated roster exports are excluded from future releases. Do not expose a server containing those private files to the Internet. A password implemented in browser JavaScript does not protect data downloaded by the page.

The plotted jitter is only a visual aid. Neighbor searches and learned distances use the unchanged feature values.
