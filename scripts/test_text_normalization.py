#!/usr/bin/env python3
"""Regression tests for spoken meaning, lexicon parsing and engine integration."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--cli', type=Path)
    p.add_argument('--bundles', type=Path, default=Path('models-v2-int8-linear-decoder'))
    args = p.parse_args()
    cases = []

    def case(text, expected, language='english', **options):
        cases.append((dict(text=text, language=language, **options), expected))

    # The user's concrete failures, checked as complete utterances.
    case('The startup secured $5.2M in venture capital, a huge leap from their initial $450K seed round.',
         'The startup secured five point two million dollars in venture capital, a huge leap from their initial four hundred fifty thousand dollars seed round.')
    case('1,450,000', 'one million four hundred fifty thousand')
    case('You can reach the hotel front desk at (212) 555-0142 ext. 402 anytime.',
         'You can reach the hotel front desk at two one two, five five five, zero one four two, extension four zero two anytime.')
    case('87M-34KEL6', 'eight seven em, three four kay ee el six')
    case('78oC', 'seventy-eight degrees Celsius')
    case('12h, 30s, 400 mbps, 45 lbs.', 'twelve hours, thirty seconds, four hundred megabits per second, forty-five pounds.')

    for text, expected in [
        ('(310) 799-1029', 'three one zero, seven nine nine, one zero two nine'),
        ('(310)799-1029', 'three one zero, seven nine nine, one zero two nine'),
        ('310.799.1029', 'three one zero, seven nine nine, one zero two nine'),
        ('310 799 1029', 'three one zero, seven nine nine, one zero two nine'),
        ('625-29-2002', 'six two five, two nine, two zero zero two'),
        ('625 - 29 - 2002', 'six two five, two nine, two zero zero two'),
    ]:
        case(text,expected,lexicon='-')

    # Dates/times precede generic identifier, range and numeric processing.
    for text, expected in [
        ('January 1, 1980', 'January first, nineteen eighty'),
        ('Jan 1, 1980', 'January first, nineteen eighty'),
        ('Jan. 1, 1980', 'January first, nineteen eighty'),
        ('January 01,1980', 'January first, nineteen eighty'),
        ('12-MAR-09', 'March twelfth, two thousand nine'),
        ('12 - Mar - 09', 'March twelfth, two thousand nine'),
        ('12 March 2009', 'March twelfth, two thousand nine'),
        ('12th March, 2009', 'March twelfth, two thousand nine'),
        ('March 12th, 2009', 'March twelfth, two thousand nine'),
        ('May 5', 'May fifth'), ('Feb. 29, 2000', 'February twenty-ninth, two thousand'),
        ('Mon Jan. 1, 1980', 'Monday, January first, nineteen eighty'),
        ('Fri, March 12, 2009', 'Friday, March twelfth, two thousand nine'),
        ('Thurs. 12 - Mar - 09', 'Thursday, March twelfth, two thousand nine'),
        ('Wed. September 30, 2026', 'Wednesday, September thirtieth, twenty twenty-six'),
        ('12-Mar-69', 'March twelfth, twenty sixty-nine'),
        ('12-Mar-70', 'March twelfth, nineteen seventy'),
        ('01/01/1980', 'one one nineteen eighty'),
        ('12-31-2026', 'twelve thirty-one twenty twenty-six'),
        ('1980-01-01', 'nineteen eighty one one'),
        ('2026 / 03 / 12', 'twenty twenty-six three twelve'),
        ('03 / 12 / 09', 'three twelve two thousand nine'),
        ('11:06P.M.', 'eleven oh six pee em.'),
        ('11:06 PM', 'eleven oh six pee em'),
        ('11 : 06 P. M.', 'eleven oh six pee em.'),
        ('01:05a.m.', 'one oh five ay em.'),
        ('12:00 PM', 'twelve pee em'), ('00:00', "zero o'clock"),
        ('23:06', 'twenty-three oh six'), ('09:30', 'nine thirty'),
        ('11:00:01 AM', 'eleven oh zero and one second ay em'),
        ('23 : 05 : 09', 'twenty-three oh five and nine seconds'),
        ('Meet Mon. Jan. 1, 1980 at 11:06 PM.', 'Meet Monday, January first, nineteen eighty at eleven oh six pee em.'),
        ('Mon and Fri', 'Mon and Fri'),
        ('31/99/2026', 'thirty-one ninety-nine twenty twenty-six'),
        ('25:99', 'twenty-five ninety-nine'),
        ('ABC/42:XY-07', 'ABC forty-two XY seven'),
        ('32-Mar-09', 'thirty-two Mar nine'),
        ('Feb 29, 1900', 'Feb twenty-nine, nineteen hundred'),
        ('February 29, 2024', 'February twenty-ninth, twenty twenty-four'),
        ('02/29/2024', 'two twenty-nine twenty twenty-four'),
        ('April 31, 2026', 'April thirty-one, twenty twenty-six'),
        ('11:60 PM', 'eleven sixty PM'),
        ('Jan 1.5', 'Jan one point five'),
        ('Marching 12, 2026', 'Marching twelve, twenty twenty-six'),
        ('At 11:06 P.M. Then leave.', 'At eleven oh six pee em. Then leave.'),
        ('At 11:06 P.M. tomorrow.', 'At eleven oh six pee em tomorrow.'),
        ('Janitor 1', 'Janitor one'),
    ]:
        case(text,expected,lexicon='-')
    case('Jan. 1, 1980 at 11:06 PM', 'Jan. 1, 1980 at 11:06 PM', enabled=False)
    case('1980-01-01 à 11:06', 'mille-neuf-cent-quatre-vingts un un à onze heures six', language='french', lexicon='-')

    # Scientific forms take priority over identifiers and year readings.
    for text, expected in [
        ('5.6 × 10^-34', 'five point six times ten to the minus thirty-four'),
        ('1.2 x 10^23', 'one point two times ten to the twenty-three'),
        ('2.7E-35', 'two point seven times ten to the minus thirty-five'),
        ('5.3e1,200', 'five point three times ten to the one thousand two hundred'),
        ('5.3e+1200', 'five point three times ten to the one thousand two hundred'),
        ('2E3', 'two times ten to the three'),
        ('-3.050e−02', 'minus three point zero five zero times ten to the minus two'),
        ('+1 x10 ^ +2026', 'plus one times ten to the two thousand and twenty-six'),
        ('(1e0), 2e0004.', '(one times ten to the zero), two times ten to the four.'),
        ('6e1,000,000', 'six times ten to the one million'),
        ('3.1×10^−9 kg', 'three point one times ten to the minus nine kg'),
        ('A2E3 2E3X', 'A two E three two E three X'),
        ('5.6 × 10^-34', 'cinq virgule six fois dix à la puissance moins trente-quatre'),
        ('2,7E-35', 'deux virgule sept fois dix à la puissance moins trente-cinq'),
        ('5,3e+1,200', 'cinq virgule trois fois dix à la puissance mille-deux-cents'),
    ]:
        case(text, expected, language='french' if 'fois' in expected else 'english', lexicon='-')
    case('2.7E-35', '2.7E-35', enabled=False)

    for a, b in [
        ('0', 'zero'), ('007', 'zero zero seven'), ('-42', 'minus forty-two'),
        ('+3.050', 'plus three point zero five zero'), ('100', 'one hundred'),
        ('1000000', 'one million'), ('1,000,001', 'one million and one'),
        ('$1,500', 'one thousand five hundred dollars'),
        ('$1.50', 'one dollar and fifty cents'), ('$0.05', 'zero dollars and five cents'),
        ('$1.00', 'one dollar'), ('$2.01', 'two dollars and one cent'),
        ('-$5', 'minus five dollars'), ('€1', 'one euro'), ('£2.01', 'two pounds and one penny'),
        ('$1M', 'one million dollars'), ('$2B', 'two billion dollars'),
        ('$14 million', 'fourteen million dollars'), ('$14 billion', 'fourteen billion dollars'),
        ('€14 million', 'fourteen million euros'), ('£14 billion', 'fourteen billion pounds'),
        ('$14 milliard', 'fourteen billion dollars'), ('$14 MILLION', 'fourteen million dollars'),
        ('$1 million', 'one million dollars'), ('$1.50 million', 'one point five zero million dollars'),
        ('14 million €', 'fourteen million euros'), ('$14 thousand', 'fourteen thousand dollars'),
        ('$14 millionaire', 'fourteen dollars millionaire'),
        ('$1.5 trillion', 'one point five trillion dollars'),
        ('$1.5T', 'one point five trillion dollars'), ('$1t', 'one trillion dollars'),
        ('£2 trillion', 'two trillion pounds'), ('€1 trillion', 'one trillion euros'),
        ('1.5 trillion €', 'one point five trillion euros'),
        ('$14 TRILLION', 'fourteen trillion dollars'),
        ('$14 trillions', 'fourteen trillion dollars'),
        ('$14 trillionaire', 'fourteen dollars trillionaire'),
        ('2026', 'twenty twenty-six'), ('1943', 'nineteen forty-three'),
        ('1000', 'one thousand'), ('1001', 'ten oh one'),
        ('1900', 'nineteen hundred'), ('1905', 'nineteen oh five'),
        ('1909', 'nineteen oh nine'), ('1910', 'nineteen ten'),
        ('2000', 'two thousand'), ('2001', 'two thousand one'),
        ('2005', 'two thousand five'), ('2009', 'two thousand nine'),
        ('1999', 'nineteen ninety-nine'), ('2010', 'twenty ten'),
        ('2020', 'twenty twenty'), ('2099', 'twenty ninety-nine'),
        ('1234', 'twelve thirty-four'), ('2100', 'two thousand one hundred'),
        ('999', 'nine hundred ninety-nine'),
        ('1,943', 'one thousand nine hundred forty-three'),
        ('2026.5', 'two thousand and twenty-six point five'),
        ('202600', 'two hundred two thousand six hundred'),
        ('$2026', 'two thousand and twenty-six dollars'),
        ('2026 kg', 'two thousand and twenty-six kilograms'),
        ('A2026', 'A two thousand and twenty-six'),
        ('1999-2026', 'nineteen ninety-nine to twenty twenty-six'),
        ('1900-2005', 'nineteen hundred to two thousand five'),
        ('78°C', 'seventy-eight degrees Celsius'), ('1ºF', 'one degree Fahrenheit'),
        ('-4oC', 'minus four degrees Celsius'),
        ('1h 1s 1 lb', 'one hour one second one pound'),
        ('400 Mbps', 'four hundred megabits per second'),
        ('8 MB/s', 'eight megabytes per second'), ('12kg', 'twelve kilograms'),
        ('10%', 'ten percent'), ('21st', 'twenty-first'), ('12th', 'twelfth'),
        ('22nd', 'twenty-second'), ('23rd', 'twenty-third'),
        ('(402)', 'four zero two'), ('A007', 'A seven'),
        ('4D', 'four D'), ('9800XT', 'nine thousand eight hundred XT'),
        ('IN2BACH', 'IN two BACH'), ('A000B', 'A zero B'),
        ('A12B34C', 'A twelve B thirty-four C'),
        ('9800xt', '9800xt'), ('in2BACH', 'in2BACH'),
        ('555-0142', 'five five five, zero one four two'),
        ('+44 20 7946 0958', 'plus four four, two zero, seven nine four six, zero nine five eight'),
        ('212-555-0142 x402', 'two one two, five five five, zero one four two, extension four zero two'),
        ('R2-D2', 'ar two, dee two'), ('12-34', 'twelve to thirty-four'),
        ('12 - 34', 'twelve to thirty-four'), ('12- 34', 'twelve to thirty-four'),
        ('12 -34', 'twelve to thirty-four'), ('1-2', 'one to two'),
        ('100-200 people', 'one hundred to two hundred people'),
        ('12-34-56', 'one two, three four, five six'),
        ('12 - 34 - 56', 'one two, three four, five six'),
        ('554-33-2998', 'five five four, three three, two nine nine eight'),
        ('554 - 33 - 2998', 'five five four, three three, two nine nine eight'),
        ('12-34-', 'one two, three four-'),
        ('ABC(123)', 'ABC one two three'), ('87(34)', 'eight seven three four'),
        ('AB_12', 'AB twelve'),
        ('150-milligrams', 'one hundred fifty milligrams'),
        ('150-mg', 'one hundred fifty milligrams'),
        ('3-D-printer', 'three dee printer'), ('3-D', 'three dee'),
        ('1.5-liter', 'one point five liter'), ('-150-milligrams', 'minus one hundred fifty milligrams'),
        ('87M34KEL6', 'eighty-seven M thirty-four KEL six'),
        ('87m34kel6', '87m34kel6'), ('iPhone15', 'iPhone15'), ('mp3', 'mp3'),
        ('vitamin-B12', 'vitamin bee one two'),
        ('1,2345', 'one,two thousand three hundred forty-five'),
        ('300K', 'three hundred kelvins'),
        ('212-555-0142-ABC', 'two one two, five five five, zero one four two, ABC'),
        ('Ordinary prose stays unchanged.', 'Ordinary prose stays unchanged.'),
        ('https://example.com/v2/123', 'https://example.com/v2/123'),
        ('mail42@example.com', 'mail42@example.com'),
        ('AI is not RAIN or xAI.', 'eigh-eye is not RAIN or xAI.'),
        ('“AI” and 45lbs—really.', '“eigh-eye” and forty-five pounds—really.'),
        ('Dr. Smith met Mrs. Jones, e.g. at MacOS.', 'Doctor Smith met Misses Jones, for example at mac-oh-ess.'),
    ]:
        case(a, b, lexicon='')

    for a, b in [
        ('0', 'zéro'), ('1 450 000', 'un-million-quatre-cent-cinquante-mille'),
        ('1\u202f450\u202f000', 'un-million-quatre-cent-cinquante-mille'),
        ('1\u00a0450\u00a0000', 'un-million-quatre-cent-cinquante-mille'),
        ('1 2345', 'un deux-mille-trois-cent-quarante-cinq'),
        ('1,05', 'un virgule zéro cinq'), ('-42', 'moins quarante-deux'),
        ('1h', 'une heure'), ('21h', 'vingt-et-une heures'),
        ('1,50 €', 'un euro et cinquante centimes'), ('5,2 €', 'cinq virgule deux euros'),
        ('€5,2M', "cinq virgule deux millions d'euros"),
        ('$14 million', 'quatorze millions de dollars'),
        ('€14 millions', "quatorze millions d'euros"),
        ('£14 milliard', 'quatorze milliards de livres sterling'),
        ('$14 milliards', 'quatorze milliards de dollars'),
        ('€14 billion', "quatorze billions d'euros"),
        ('€1,5 trillion', "un virgule cinq trillions d'euros"),
        ('€1 trillion', "un trillion d'euros"),
        ('£1 trillion', 'un trillion de livres sterling'),
        ('€1T', "un billion d'euros"),
        ('1,5 trillions €', "un virgule cinq trillions d'euros"),
        ('2026', 'deux-mille-vingt-six'),
        ('4D', 'quatre D'), ('IN2BACH', 'IN deux BACH'),
        ('14 millions €', "quatorze millions d'euros"),
        ('€1 million', "un million d'euros"), ('€14 mille', 'quatorze mille euros'),
        ('12-34', 'douze à trente-quatre'), ('12 - 34', 'douze à trente-quatre'),
        ('554-33-2998', 'cinq cinq quatre, trois trois, deux neuf neuf huit'),
        ('$1K', 'mille dollars'),
        ('78oC', 'soixante-dix-huit degrés Celsius'),
        ('400 mbps', 'quatre-cents mégabits par seconde'),
        ('1er', 'premier'), ('1re', 'première'), ('2e', 'deuxième'),
        ('80', 'quatre-vingts'), ('81', 'quatre-vingt-un'), ('200', 'deux-cents'),
        ('01 23 45 67 89 poste 402', 'zéro un, deux trois, quatre cinq, six sept, huit neuf, poste quatre zéro deux'),
        ('M. Dupont et Mme Martin.', 'Monsieur Dupont et Madame Martin.'),
        ('Mme. Martin et Dr. Dupont.', 'Madame Martin et Docteur Dupont.'),
        ('Un café ☕ et 2 croissants.', 'Un café ☕ et deux croissants.'),
        ('150-milligrammes', 'cent-cinquante milligrammes'),
        ('3-D-imprimante', 'trois dé imprimante'),
        ('«M. Dupont» attend 1h.', '«Monsieur Dupont» attend une heure.'),
    ]:
        case(a, b, 'french', lexicon='')

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        csv = root/'custom.csv'
        csv.write_text('\ufeffsource,replacement\r\n# comment\r\n"A.I.","eigh, eye"\r\nAI,A.I.\r\nAI Lab,the lab\r\n"say ""hi""",hello\r\nremove,\r\ncafé,coffee\r\n', encoding='utf-8')
        case('AI Lab AI A.I. RAIN xAI say "hi" remove café décaféiné',
             'A.I. Lab A.I. eigh, eye RAIN xAI hello  coffee décaféiné', lexicon=str(csv))
        case('$5 Dr. 87M', '$5 Dr. 87M', enabled=False, lexicon=str(root/'missing.csv'))
        case('Dr.', 'Dr.', lexicon='-')
        case('FY2026', 'fiscal year twenty twenty-six', lexicon='')
        insensitive = root/'literal-insensitive.csv'
        insensitive.write_text('source,replacement\n/iMr.,Mister\n/ia+b,plus word\n/iCafé,coffee\n/iŒuvre,work\n/iStraße,street\n/iCash,$$\n/i[foo],bracket\n',encoding='utf-8')
        case('mr. MR. Mr. amr. A+B aaab CASH [FOO]', 'Mister Mister Mister amr. plus word aaab $$ bracket',lexicon=str(insensitive))
        case('café CAFÉ Café ŒUVRE œuvre STRASSE Straße', 'coffee coffee coffee work work street street',lexicon=str(insensitive),language='french')
        case('café décaféiné', 'coffee décaféiné',lexicon=str(insensitive))
        slash = root/'literal-slash.csv'
        slash.write_text('source,replacement\n//foo.bar,slash word\n///foo,slash pair\n//rfoo,escaped prefix\n//,slash\n/i/case,insensitive slash\n')
        case('/foo.bar /Foo.bar /fooXbar //foo /rfoo /rFoo / /CASE', 'slash word /Foo.bar /fooXbar slash pair escaped prefix /rFoo slash insensitive slash',lexicon=str(slash))
        # Sensitive regex keeps its case; insensitive regex retains capture case.
        sensitive = root/'regex-sensitive.csv'
        sensitive.write_text('source,replacement\n/r\\b(cat)(dog)\\b,$2 $1\n')
        case('catdog Catdog CATDOG', 'dog cat Catdog CATDOG',lexicon=str(sensitive))
        ci_regex = root/'regex-insensitive.csv'
        ci_regex.write_text('source,replacement\n/R\\b(cat)(dog)\\b,$2 $1\n/R^start|end$,edge\n/R\\bFY(20\\d\\d)\\b,fiscal year $1\n')
        case('CatDOG CATdog', 'DOG Cat dog CAT',lexicon=str(ci_regex))
        case('START start end END', 'edge start end edge',lexicon=str(ci_regex))
        case('fy2026 fY2027 xFY2026', 'fiscal year twenty twenty-six fiscal year twenty twenty-seven xFY2026',lexicon=str(ci_regex))
        for index, data, expected in [
            (0, 'AI,exact\n/iAI,insensitive\n', 'exact insensitive insensitive'),
            (1, '/iAI,insensitive\nAI,exact\n', 'insensitive insensitive insensitive'),
        ]:
            path=root/f'case-precedence{index}.csv';path.write_text(data)
            case('AI ai Ai',expected,lexicon=str(path))
        for index, source in enumerate(['/i','/xhello','/']):
            path=root/f'bad-prefix{index}.csv';path.write_text('source,replacement\n'+source+',bad\n')
            case('hello',{'error':'lexicon'},lexicon=str(path))
        regex_csv = root/'regex.csv'
        regex_csv.write_text('source,replacement\n/r\\bFY(20\\d\\d)\\b,fiscal year $1\n',encoding='utf-8')
        case('FY2026 and FY2027.', 'fiscal year twenty twenty-six and fiscal year twenty twenty-seven.', lexicon=str(regex_csv))
        case('xFY2026 FY2026x FY2026_', 'xFY2026 FY2026x FY two thousand and twenty-six_', lexicon=str(regex_csv))
        case('(FY2026)', '(fiscal year twenty twenty-six)', lexicon=str(regex_csv))
        case('FY2026', 'FY2026', enabled=False, lexicon=str(regex_csv))
        # Regex boundaries are explicit; literal entries retain implicit ones.
        regex_csv2 = root/'regex-substrings.csv'
        regex_csv2.write_text('source,replacement\n/rcat,dog\ncatfish,fish\n',encoding='utf-8')
        case('cat scatter catfish', 'dog sdogter dogfish', lexicon=str(regex_csv2))
        # First matching CSV row wins, even if a later rule could match more.
        for index, data, expected in [
            (0, '/rf(o+),regex\nfoo,literal\n', 'regex regex'),
            (1, 'foo,literal\n/rf(o+),regex\n', 'literal regex'),
            (2, '/rf(o),short\nfoo,long\n', 'shorto'),
        ]:
            path = root/f'precedence{index}.csv'; path.write_text(data)
            case('foo fooo' if index != 2 else 'foo',expected,lexicon=str(path))
        for index, data, expected in [
            (0, 'AI,short\nAI Lab,long\n', 'short Lab'),
            (1, 'AI Lab,long\nAI,short\n', 'long'),
            (2, 'cat,literal\n/rcatfish,regex\n', 'regex'),
            (3, '/r^,prefix \ncat,literal\n', 'prefix cat'),
            (4, 'cat,literal\n/r^,prefix \n', 'literal'),
        ]:
            path = root/f'first-match{index}.csv'; path.write_text(data)
            case('AI Lab' if index < 2 else 'catfish' if index == 2 else 'cat',expected,lexicon=str(path))
        features = root/'regex-features.csv'
        features.write_text('source,replacement\n/r^start,begin\n/rend$,finish\n/r\\b(cat)(dog)\\b,$2 $1\n/r\\bwhole\\b,[$&]\n/rcash,$$\nliteral,$$\n/rdrop,\n',encoding='utf-8')
        case('start start catdog whole cash literal drop end end', 'begin start dog cat [whole] $ $$  end finish',lexicon=str(features))
        nonrecursive = root/'regex-nonrecursive.csv'
        nonrecursive.write_text('source,replacement\n/rfoo,bar\nbar,baz\n')
        case('foo bar', 'bar baz',lexicon=str(nonrecursive))
        for index, pattern, text, expected in [
            (0, '^foo|bar', 'foo foo bar', 'matched foo matched'),
            (1, r'\^foo', '^foo x^foo', 'matched xmatched'),
            (2, '[^^]at', 'cat ^at', 'matched ^at'),
            (3, '(?=^foo)foo', 'foo foo', 'matched foo'),
            (4, '[[:alpha:]^]at', 'cat ^at', 'matched matched'),
        ]:
            path = root/f'anchors{index}.csv'; path.write_text('source,replacement\n/r'+pattern+',matched\n')
            case(text,expected,lexicon=str(path))
        comma = root/'regex-comma.csv'
        comma.write_text('source,replacement\n"/r\\b([A-Z]{2,3})-(\\d+)\\b",$1 number $2\n')
        case('AB-42', 'AB number forty-two',lexicon=str(comma))
        zero = root/'regex-zero.csv'
        zero.write_text('source,replacement\n/r^,begin \n/r$, end\n/r(?=cat),big \n')
        case('cat cat', 'begin cat big cat end',lexicon=str(zero))
        case('', 'begin ',lexicon=str(zero)) # tie: first zero-width rule wins
        utf = root/'regex-utf.csv'
        utf.write_text('source,replacement\n/rcafé,coffee\n')
        case('Un café ☕.', 'Un coffee ☕.',lexicon=str(utf))
        broken = root/'regex-broken-utf.csv'
        broken.write_text('source,replacement\n/r(.).,$1\n')
        case('é', {'error':'invalid UTF-8'}, lexicon=str(broken))
        for index, pattern in enumerate(['/r(', '/r', '/R', '/r(?<=x)y']):
            path = root/f'invalid-regex{index}.csv'; path.write_text('source,replacement\n'+pattern+',bad\n')
            case('hello', {'error':'lexicon regex'}, lexicon=str(path))
        case('hello', {'error': 'Cannot read lexicon'}, lexicon=str(root/'missing.csv'))
        for index, data in enumerate(['source,replacement\na,b\na,c\n', 'source,replacement\na,b,c\n', 'source,replacement\n"a,b\n', 'source,replacement\n"a"oops,b\n']):
            path = root/f'bad{index}.csv'
            path.write_text(data)
            case('hello', {'error': ''}, lexicon=str(path))

        run = subprocess.run([str(args.probe.resolve())], input=''.join(json.dumps(q)+'\n' for q, _ in cases),
                             capture_output=True, text=True, check=True, timeout=30)
        lines = run.stdout.splitlines()
        assert len(lines) == len(cases), run.stdout
        for (q, expected), line in zip(cases, lines):
            actual = json.loads(line)
            if isinstance(expected, dict):
                assert 'error' in actual and expected['error'] in actual['error'], (q, actual)
            else:
                assert actual == {'text': expected}, (q, expected, actual)
        print(f'{len(cases)} normalization/CSV cases passed')

    if args.cli:
        def inspect(text, *options, language='english'):
            command = [str(args.cli.resolve()), '--models-dir', str(args.bundles/language),
                       '--inspect-text', *options, text]
            return json.loads(subprocess.check_output(command, text=True))
        result = inspect('Dr. Smith paid $1,500 for 12h.')
        assert result['normalized_text'] == 'Doctor Smith paid one thousand five hundred dollars for twelve hours.'
        assert len(result['chunks']) == 1
        assert 'Doctor Smith' in result['chunks'][0]['text']
        result = inspect('M. Dupont attend 1h.', language='french')
        assert result['normalized_text'] == 'Monsieur Dupont attend une heure.'
        assert len(result['chunks']) == 1
        result = inspect('Dr. Smith paid $1,500.', '--no-normalize-text')
        assert result['normalized_text'] == 'Dr. Smith paid $1,500.'
        result = inspect('Dr. Smith filed FY2026.')
        assert result['normalized_text'] == 'Doctor Smith filed fiscal year twenty twenty-six.'
        assert len(result['chunks']) == 1
        result = inspect('Meet Mon. Jan. 1, 1980 at 11:06 PM.')
        assert result['normalized_text'] == 'Meet Monday, January first, nineteen eighty at eleven oh six pee em.'
        assert len(result['chunks']) == 1
        result = inspect('4D 9800XT IN2BACH')
        assert result['normalized_text'] == 'four D nine thousand eight hundred XT IN two BACH'
        print('English/French chunking and bypass integration passed')


if __name__ == '__main__':
    main()
