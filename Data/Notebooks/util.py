

class Country:
    def __init__(self, name, code_2L, code_3L) -> None:
        self.NAME = name
        self.CODE_2L = code_2L # 2 letter code
        self.CODE_3L = code_3L # 3 letter code


class COUNTRIES:
    ANDORRA = Country('Andorra', 'AD', 'AND')
    ALBANIA = Country('Albania', 'AL', 'ALB')
    ARMENIA = Country('Armenia', 'AM', 'ARM')
    AUSTRIA = Country('Austria', 'AT', 'AUT')
    AZERBAIJAN = Country('Azerbaijan', 'AZ', 'AZE')
    BELGIUM = Country('Belgium', 'BE', 'BEL')
    BULGARIA = Country('Bulgaria', 'BG', 'BGR')
    BELARUS = Country('Belarus', 'BY', 'BLR')
    SWITZERLAND = Country('Switzerland', 'CH', 'CHE')
    CYPRUS = Country('Cyprus', 'CY', 'CYP')
    CZECHIA = Country('Czechia', 'CZ', 'CZE')
    GERMANY = Country('Germany', 'DE', 'DEU')
    DENMARK = Country('Denmark', 'DK', 'DNK')
    ESTONIA = Country('Estonia', 'EE', 'EST')
    GREECE = Country('Greece', 'GR', 'GRC')
    SPAIN = Country('Spain', 'ES', 'ESP')
    FINLAND = Country('Finland', 'FI', 'FIN')
    FRANCE = Country('France', 'FR', 'FRA')
    GEORGIA = Country('Georgia', 'GE', 'GEO')
    CROATIA = Country('Croatia', 'HR', 'HRV')
    HUNGARY = Country('Hungary', 'HU', 'HUN')
    IRELAND = Country('Ireland', 'IE', 'IRL')
    ICELAND = Country('Iceland', 'IS', 'ISL')
    ITALY = Country('Italy', 'IT', 'ITA')
    LIECHTENSTEIN = Country('Liechtenstein', 'LI', 'LIE')
    LITHUANIA = Country('Lithuania', 'LT', 'LTU')
    LUXEMBOURG = Country('Luxembourg', 'LU', 'LUX')
    LATVIA = Country('Latvia', 'LV', 'LVA')
    MOLDOVA = Country('Moldova', 'MD', 'MDA')
    MONTENEGRO = Country('Montenegro', 'ME', 'MNE')
    NORTH_MACEDONIA = Country('North Macedonia', 'MK', 'MKD')
    MALTA = Country('Malta', 'MT', 'MLT')
    NETHERLANDS = Country('Netherlands', 'NL', 'NLD')
    NORWAY = Country('Norway', 'NO', 'NOR')
    POLAND = Country('Poland', 'PL', 'POL')
    PORTUGAL = Country('Portugal', 'PT', 'PRT')
    ROMANIA = Country('Romania', 'RO', 'ROU')
    SERBIA = Country('Serbia', 'RS', 'SRB')
    SWEDEN = Country('Sweden', 'SE', 'SWE')
    SLOVENIA = Country('Slovenia', 'SI', 'SVN')
    SLOVAKIA = Country('Slovakia', 'SK', 'SVK')



def get_country_names():
    return [country.NAME for country in COUNTRIES.__dict__.values() if isinstance(country, Country)]

def get_country_codes():
    return [country.CODE_2L for country in COUNTRIES.__dict__.values() if isinstance(country, Country)]

def get_country_codes_3L():
    return [country.CODE_3L for country in COUNTRIES.__dict__.values() if isinstance(country, Country)]