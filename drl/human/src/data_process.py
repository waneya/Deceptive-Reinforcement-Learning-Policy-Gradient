from collections import Counter
import numpy as np
import csv


class Question:
    def __init__(self, q_id, model, map_name, num_choice, density, corr_idx, survey_idx):
        self.q_id = q_id
        self.model = model
        self.map_name = map_name
        self.num_choice = num_choice
        self.density = density
        self.corr_idx = corr_idx
        self.survey_idx = survey_idx

    def __str__(self):
        return '{0:s}: {1:d}, {2:s}, {3:d}, {4:d}, {5:d}, {6:d}' \
            .format(self.q_id, self.model, self.map_name, self.num_choice,
                    self.density, self.corr_idx, self.survey_idx)


class Answer:
    def __init__(self, question, rates):
        self.question = question
        if len(rates) != question.num_choice:
            pass  # error msg here
        self.rates = rates
        self.real_prob = rates[question.corr_idx - 1] / sum(rates)  # corr_idx is 1-based
        self.correctness = 0
        most_star = 0
        for r in rates:
            most_star = max(most_star, r)
        most_count = 0
        for r in rates:
            if r == most_star:
                most_count += 1
        if rates[question.corr_idx - 1] == most_star:
            self.correctness = 1 / most_count


class Report:
    def __init__(self, start_time, end_time, ip_address, duration, mturk_id, gender, age, random_id):
        self.start_time = start_time
        self.end_time = end_time
        self.ip_address = ip_address
        self.duration = duration
        self.mturk_id = mturk_id
        self.gender = gender
        self.age = age
        self.random_id = random_id
        self.answers = list()
        self.correctness = 0

    def addAnswer(self, answer):
        self.answers.append(answer)
        self.correctness += answer.correctness


def readSurveyInfo():
    info_file = '../data/survey_info.csv'
    q_map = dict()
    with open(info_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for survey_idx in range(1, 13):
            line = next(reader)
            if int(line[0]) != survey_idx:
                print('unmatched survey index')
            line = next(reader)
            num_question = int(line[0])
            for _ in range(num_question):
                line = next(reader)
                if line[0] == '259':
                    print('skipped Q259')
                    continue
                q_id = 'Q' + line[0]
                model = int(line[1])
                map_name = line[2]
                num_choice = int(line[3]) - 1
                density = int(line[4])
                corr_idx = int(line[5])
                q = Question(q_id, model, map_name, num_choice, density, corr_idx, survey_idx)
                q_map[q_id] = q
    return q_map


def processData(questions):
    data_file = '../data/12.csv'
    a_list = list()
    r_map = dict()
    with open(data_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        # print(header)
        for i, r in enumerate(reader):
            # print(r)
            mturk_id = r[17]
            kwargs = {'start_time': r[0], 'end_time': r[1],
                      'ip_address': r[3], 'duration': int(r[5]),
                      'mturk_id': mturk_id, 'gender': r[18] + r[19],
                      'age': int(r[20]), 'random_id': r[-8]}
            report = Report(**kwargs)
            a_idx = 21
            while a_idx < len(r):
                q_name = header[a_idx]
                # print(q_name)
                if not q_name.startswith('Q'):
                    break
                if q_name.startswith('Q259'):
                    a_idx += 1
                    continue
                q_name_split = q_name.split('_')
                if q_name_split[1] != '1':
                    print('question does not align')
                    raise
                q = questions[q_name_split[0]]
                # print(q)
                rates = []
                answered = False
                for _ in range(q.num_choice):
                    r_str = r[a_idx]
                    if r_str == '':
                        rates.append(0)
                    else:
                        answered = True
                        rates.append(int(r[a_idx]))
                    a_idx += 1
                if answered:
                    # print(rates)
                    a = Answer(q, rates)
                    report.addAnswer(a)
                    a_list.append(a)
            print('finished report', i)
            if mturk_id in r_map:
                print('{:s} already in map'.format(mturk_id))
            r_map[mturk_id] = report
    return a_list, r_map


def getTopReports(reports, fraction=0.2):
    for r in reports:
        pass
    pass


def genTable(ans, num_choice):
    tbl = list()
    for m in [0, 1, 2, 3]:
        p = list()
        for d in [0, 1, 2]:
            k = (m, num_choice, d)
            l = ans[k]
            p.append(sum(l) / len(l))
        tbl.append(p)
    return tbl


def printTbl(tbl):
    out = ''
    for r in tbl:
        for c in r:
            out += '{:.5f} '.format(c)
        out += '\n'
    print(out)


def getPlotData(answers):
    prob_ans = dict()
    corr_ans = dict()
    for a in answers:
        q = a.question
        model = q.model
        map_name = q.map_name
        num_choice = q.num_choice
        density = q.density
        k = (model, num_choice, density)
        if k not in prob_ans:
            prob_ans[k] = list()
        prob_ans[k].append(a.real_prob)
        if k not in corr_ans:
            corr_ans[k] = list()
        corr_ans[k].append(a.correctness)
    prob_tbl_3 = genTable(prob_ans, 3)
    prob_tbl_5 = genTable(prob_ans, 5)
    corr_tbl_3 = genTable(corr_ans, 3)
    corr_tbl_5 = genTable(corr_ans, 5)
    printTbl(prob_tbl_3)
    printTbl(prob_tbl_5)
    printTbl(corr_tbl_3)
    printTbl(corr_tbl_5)


def demographic(reports):
    male = 0
    female = 0
    other = 0
    ages = list()
    for k, r in reports.items():
        if r.gender == 'Male':
            male += 1
        elif r.gender == 'Female':
            female += 1
        else:
            other += 1
        ages.append(r.age)
    print('Male: {0:d}, Female: {1:d}, Other: {2:d}'.format(male, female, other))
    print('Average age: {0:.2f}'.format(np.mean(ages)))
    print('Total: {:d}'.format(len(ages)))


if __name__ == '__main__':
    questions = readSurveyInfo()
    # for k, v in questions.items():
    # 	print(k, v)
    answers, reports = processData(questions)
    # getTopReports(reports, 0.2)
    getPlotData(answers)
    demographic(reports)
