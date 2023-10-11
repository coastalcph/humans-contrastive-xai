# Aligning the tokenization of different language model tokenizers with the tokenization in the eye-tracking corpora is really tricky.
# We did our best to account for as many cases as possible.
# Some cases are so specific that they would need to be hard-coded.
# For example, the ZUCO corpus contains a few instances of "U.S" which is seen as a single token but separated by most tokenizers.
# We decided to simply ignore these very specific cases but encourage you to do better.
import numpy as np


def merge_subwords(tokens, summed_importance, pooling='max'):
    adjusted_tokens = []
    adjusted_importance = []

    current_token = ""
    current_importance = 0

    # Tokenizers use different word piece separators. We simply check for both here
    word_piece_separators = ("##", "_")
    for i, token in enumerate(tokens):
        # We sum the importance of word pieces
        if pooling == 'max':
            current_importance = current_importance if current_importance>summed_importance[i] else summed_importance[i]
        else:
            current_importance += summed_importance[i]

        # Identify word piece
        if token.startswith(word_piece_separators):
            #skip the hash tags
            current_token += token[2:]

        else:
            current_token += token


        # Is this the last token of the sentence?
        if i == len(tokens)-1:
            adjusted_tokens.append(current_token)
            adjusted_importance.append(current_importance)

        else:
        # Are we at the end of a word?
            if not tokens[i+1].startswith(word_piece_separators):
                # append merged token and importance
                adjusted_tokens.append(current_token)
                adjusted_importance.append(current_importance)

                # reset
                current_token = ""
                current_importance = 0
    return adjusted_tokens, adjusted_importance

# Word piece tokenization splits words separated by hyphens. Most eye-tracking corpora don't do this.
# This method sums the importance for tokens separated by hyphens.
def merge_hyphens(tokens, importance, pooling=None):
    adjusted_tokens = []
    adjusted_importance = []
    hyphens = ["-", "/"]
    exceptions = ['themovieisvirtuallywithoutcontext--journalisticorhistorical.',
                  "adamsandler'seightcrazynightsgrowsonyou--likearash.",
                  "itisn'tthatstealingharvardisahorriblemovie--ifonlyitwerethatgrandafailure!",
                  'itisintenselypersonalandyet--unlikequills--deftlyshowsusthetemperofthetimes.',
                  'whyhewasgivenfreereignoverthisproject--hewrote,directed,starredandproduced--isbeyondme.',
                  "thefilmrehashesseveraloldthemesandiscappedwithpointlessextremes--it'sinsanelyviolentandverygraphic.",
                  'thereisarefreshingabsenceofcynicisminstuartlittle2--quiteararity,eveninthefamilyfilmmarket.',
                  'devosandcasselhavetremendouschemistry--theirsexualandromantictension,whileneverreallyvocalized,ispalpable.',
                  'azombiemovieineverysenseoftheword--mindless,lifeless,meandering,loud,painful,obnoxious.',
                  'thankslargelytowilliams,alltheinterestingdevelopmentsareprocessedin60minutes--therestisjustanoverexposedwasteoffilm.',
                  'inxxx,dieselisthatrarecreature--anactionherowithtablemanners,andonewhoprovesthateleganceismorethantattoodeep.',
                  "athoroughlyawfulmovie--dumb,narrativelychaotic,visuallysloppy...aweirdamalgamof`thething'andageriatric`scream.'",
                  "theonlyentertainmentyou'llderivefromthischoppyandsloppyaffairwillbefromunintentionalgiggles--severalofthem.",
                  'thewaycoppolaprofesseshisloveformovies--bothcolorfulpopjunkandtheclassicsthatunequivocallyqualifyasart--isgiddilyentertaining.',
                  "methodical,measured,andgentlytediousinitscomedy,secretballotisapurposefullyreductivemovie--whichmaybewhyit'ssosuccessfulatlodgingitselfinthebrain.",
                  'thefilmjustmightturnonmanypeopletoopera,ingeneral,anartformatoncevisceralandspiritual,wonderfullyvulgarandsublimelylofty--andasemotionallygrandaslife.',
                  'inadditiontoscoringhighfororiginalityofplot--puttingtogetherfamiliarthemesoffamily,forgivenessandloveinanewway--lilo&stitchhasanumberofotherassetstocommendittomovieaudiencesbothinnocentandjaded.'
                  ]
    if any(el in hyphens for el in tokens) and not "".join(tokens) in exceptions:
        # Get all indices of -
        indices = [i for i, x in enumerate(tokens) if x in hyphens]
        if "".join(tokens) in [
            "it'sahead-turner--thoughtfullywritten,beautifullyreadand,finally,deeplyhumanizing.",
            'afirst-class,thoroughlyinvolvingbmoviethateffectivelycombinestwosurefire,belovedgenres--theprisonflickandthefightfilm.',
            "meninblackiiachievesultimateinsignificance--it'sthesci-ficomedyspectacleaswhiffle-ballepic.",
            "theactingisstiff,thestorylacksalltraceofwit,thesetslookliketheywereborrowedfromgilligan'sisland--andthecgiscoobymightwellbetheworstspecial-effectscreationoftheyear.",
            "thisodd,poeticroadmovie,spikedbyjoltsofpopmusic,prettymuchtakesplaceinmorton'sever-watchfulgaze--andit'satributetotheactress,andtoherinventivedirector,thatthejourneyissuchamesmerizingone."
        ]:
            # import pdb;
            # pdb.set_trace()
            if len(tokens) in [21, 27, 53]:
                indices = [indices[0]]
            elif len(tokens) == 24:
                indices = indices[-2:]
            elif len(tokens) == 44:
                indices = [indices[-1]]
            else:
                indices = []
        i = 0
        while i < len(tokens):
            if i+1 in indices and i+2<len(tokens):
                combined_token = tokens[i] + tokens[i+1] + tokens[i+2]
                if pooling=='max':
                    combined_heat = np.max([importance[i], importance[i + 1], importance[i + 2]])
                else:
                    combined_heat = importance[i] + importance[i + 1] + importance[i + 2]
                i += 3
                adjusted_tokens.append(combined_token)
                adjusted_importance.append(combined_heat)
            elif i in indices:
                adjusted_tokens[-1] += tokens[i] + tokens[i+1]
                if pooling=='max':
                    adjusted_importance[-1] = np.max([adjusted_importance[-1], importance[i], importance[i + 1]])
                else:
                    adjusted_importance[-1] += importance[i] + importance[i + 1]

                i += 2
            else:
                adjusted_tokens.append(tokens[i])
                adjusted_importance.append(importance[i])
                i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance

# Word piece tokenization splits parentheses and currency symbols as separate tokens. This is not done in Zuco.

def merge_symbols(tokens, importance, pooling='max'):

    if "***off" in "".join(tokens):
        initial_symbols = ["(", "$", "€", "\"", "`", "``", "*", "-"]
    elif '' in "".join(tokens):
        initial_symbols = ["(", "$", "€", "\"", "`", "``", "*", "**", "-"]
    else:
        initial_symbols = ["(", "$", "€", "\"", "\'", "`", "``", "*", "-"]
    # if 'gollum' in tokens:
    #     end_symbols = [")", "%", "\"", "\'", ".", "!", ",", "s", "t"]
    # else:
    end_symbols = [")", "%", "\"", "\'", ".", "!", ",", "s", "t", "re", "m", "?", ":", "1", "..", "d", ';', 's,', 's.', 'n', "ll", '-the-time']
    all_symbols = initial_symbols + end_symbols
    # First check if anything needs to be done
    if any(token in all_symbols for token in tokens):
        adjusted_tokens = []
        adjusted_importance = []
        i = 0
        while i <= len(tokens)-1:
            combined_token = tokens[i]
            combined_heat = importance[i]

            # Nothing to be done for the last token
            if i <=len(tokens)-2:

                # Glue the parentheses back to the token
                if tokens[i] in initial_symbols:
                    combined_token = combined_token + tokens[i+1]
                    if pooling=='max':
                        combined_heat = np.max([combined_heat, importance[i + 1]])
                    else:
                        combined_heat = combined_heat + importance[i+1]
                    i+=1


                if i < len(tokens)-1 and tokens[i + 1] in end_symbols:
                    combined_token = combined_token + tokens[i + 1]
                    if pooling == 'max':
                        combined_heat = np.max([combined_heat, importance[i + 1]])
                    else:
                        combined_heat = combined_heat + importance[i + 1]
                    i += 1
            adjusted_tokens.append(combined_token)
            adjusted_importance.append(combined_heat)
            i += 1

        return adjusted_tokens, adjusted_importance

    else:
        return tokens, importance

def merge_albert_tokens(tokens, importance, begin_token, pooling='max'):
    adjusted_tokens = []
    adjusted_importance = []
    i = 0
    # We ignore the last token [SEP]
    # import ipdb;ipdb.set_trace()
    while i < len(tokens) - 1:
        combined_token = tokens[i]
        combined_heat = importance[i]
        # Nothing to be done for the last token
        if i < (len(tokens) -2):
            while not tokens[i+1].startswith(begin_token) and not tokens[i] == '<s>':
                combined_token = combined_token + tokens[i+1]
                if pooling == 'max':
                    combined_heat = np.max(combined_heat, importance[i + 1])
                else:
                    combined_heat = combined_heat + importance[i + 1]
                i += 1
                if i == len(tokens) - 2:
                    break
        adjusted_tokens.append(combined_token.replace(begin_token, ""))
        adjusted_importance.append(combined_heat)
        i += 1
    # Add the last token
    adjusted_tokens.append(tokens[i])
    adjusted_importance.append(importance[i])
    return adjusted_tokens, adjusted_importance