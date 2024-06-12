import os

def extract_information(file_path):
    relevant_headings = ["HISTORY OF PRESENT ILLNESS ",
                        "HOSPITAL COURSE "
                        ]
    extracted_info = {}

    with open(file_path, 'r') as file:
        current_heading = None
        for line in file:
            line = line.strip()
            if line.endswith(':'):
                current_heading = line[:-1].upper()  # Convert to uppercase for case insensitivity
                if current_heading in map(str.upper, relevant_headings):
                    extracted_info[current_heading] = ''
            elif current_heading in map(str.upper, relevant_headings):
                extracted_info[current_heading] += line + ' '

    # Combine all extracted information into a single paragraph
    paragraph = ''
    for heading, info in extracted_info.items():
        paragraph += info.strip() + "\n\n"

    return paragraph

def read_reports_from_folder(folder_path):
    reports = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            report = extract_information(file_path)
            if len(report) > 50:
                report = report.replace(" '", "'")
                reports.append(report)
    return reports
