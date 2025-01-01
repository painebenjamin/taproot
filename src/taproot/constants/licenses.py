__all__ = [
    "LICENSE_MIT",
    "LICENSE_APACHE",
    "LICENSE_MPL",
    "LICENSE_OPENRAIL",
    "LICENSE_OPENRAILPP",
    "LICENSE_CC0",
    "LICENSE_CC_BY_1",
    "LICENSE_CC_BY_SA_1",
    "LICENSE_CC_BY_ND_1",
    "LICENSE_CC_BY_NC_1",
    "LICENSE_CC_BY_NC_SA_1",
    "LICENSE_CC_BY_ND_NC_1",
    "LICENSE_CC_BY_2",
    "LICENSE_CC_BY_SA_2",
    "LICENSE_CC_BY_ND_2",
    "LICENSE_CC_BY_NC_2",
    "LICENSE_CC_BY_NC_SA_2",
    "LICENSE_CC_BY_NC_ND_2",
    "LICENSE_CC_BY_25",
    "LICENSE_CC_BY_SA_25",
    "LICENSE_CC_BY_ND_25",
    "LICENSE_CC_BY_NC_25",
    "LICENSE_CC_BY_NC_SA_25",
    "LICENSE_CC_BY_NC_ND_25",
    "LICENSE_CC_BY_3",
    "LICENSE_CC_BY_SA_3",
    "LICENSE_CC_BY_ND_3",
    "LICENSE_CC_BY_NC_3",
    "LICENSE_CC_BY_NC_SA_3",
    "LICENSE_CC_BY_NC_ND_3",
    "LICENSE_CC_BY_4",
    "LICENSE_CC_BY_SA_4",
    "LICENSE_CC_BY_ND_4",
    "LICENSE_CC_BY_NC_4",
    "LICENSE_CC_BY_NC_SA_4",
    "LICENSE_CC_BY_NC_ND_4",
    "LICENSE_CC_BY",
    "LICENSE_CC_BY_SA",
    "LICENSE_CC_BY_ND",
    "LICENSE_CC_BY_NC",
    "LICENSE_CC_BY_NC_SA",
    "LICENSE_CC_BY_NC_ND",
    "LICENSE_URLS",
    "LICENSE_NAMES",
    "LICENSE_ALLOWANCES"
]

"""License Constants"""

LICENSE_MIT = "mit"
LICENSE_APACHE = "apache"
LICENSE_MPL = "mpl"
LICENSE_OPENRAIL = "openrail"
LICENSE_OPENRAILPP = "openrail++"
LICENSE_CC0 = "cc0"
LICENSE_CC_BY_1 = "cc-by-1"
LICENSE_CC_BY_SA_1 = "cc-by-sa-1"
LICENSE_CC_BY_ND_1 = "cc-by-nd-1"
LICENSE_CC_BY_NC_1 = "cc-by-nc-1"
LICENSE_CC_BY_NC_SA_1 = "cc-by-nc-sa-1"
LICENSE_CC_BY_ND_NC_1 = "cc-by-nd-nc-1"
LICENSE_CC_BY_2 = "cc-by-2"
LICENSE_CC_BY_SA_2 = "cc-by-sa-2"
LICENSE_CC_BY_ND_2 = "cc-by-nd-2"
LICENSE_CC_BY_NC_2 = "cc-by-nc-2"
LICENSE_CC_BY_NC_SA_2 = "cc-by-nc-sa-2"
LICENSE_CC_BY_NC_ND_2 = "cc-by-nc-nd-2"
LICENSE_CC_BY_25 = "cc-by-25"
LICENSE_CC_BY_SA_25 = "cc-by-sa-25"
LICENSE_CC_BY_ND_25 = "cc-by-nd-25"
LICENSE_CC_BY_NC_25 = "cc-by-nc-25"
LICENSE_CC_BY_NC_SA_25 = "cc-by-nc-sa-25"
LICENSE_CC_BY_NC_ND_25 = "cc-by-nc-nd-25"
LICENSE_CC_BY_3 = "cc-by-3"
LICENSE_CC_BY_SA_3 = "cc-by-sa-3"
LICENSE_CC_BY_ND_3 = "cc-by-nd-3"
LICENSE_CC_BY_NC_3 = "cc-by-nc-3"
LICENSE_CC_BY_NC_SA_3 = "cc-by-nc-sa-3"
LICENSE_CC_BY_NC_ND_3 = "cc-by-nc-nd-3"
LICENSE_CC_BY_4 = "cc-by-4"
LICENSE_CC_BY_SA_4 = "cc-by-sa-4"
LICENSE_CC_BY_ND_4 = "cc-by-nd-4"
LICENSE_CC_BY_NC_4 = "cc-by-nc-4"
LICENSE_CC_BY_NC_SA_4 = "cc-by-nc-sa-4"
LICENSE_CC_BY_NC_ND_4 = "cc-by-nc-nd-4"
# Aliases
LICENSE_CC_BY = LICENSE_CC_BY_4
LICENSE_CC_BY_SA = LICENSE_CC_BY_SA_4
LICENSE_CC_BY_ND = LICENSE_CC_BY_ND_4
LICENSE_CC_BY_NC = LICENSE_CC_BY_NC_4
LICENSE_CC_BY_NC_SA = LICENSE_CC_BY_NC_SA_4
LICENSE_CC_BY_NC_ND = LICENSE_CC_BY_NC_ND_4

"""License Metadata"""
LICENSE_URLS = {
    LICENSE_MIT: "https://opensource.org/licenses/MIT",
    LICENSE_APACHE: "https://www.apache.org/licenses/LICENSE-2.0",
    LICENSE_MPL: "https://www.mozilla.org/en-US/MPL/2.0/",
    LICENSE_OPENRAIL: "https://bigscience.huggingface.co/blog/bigscience-openrail-m",
    LICENSE_OPENRAILPP: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
    # CC0
    LICENSE_CC0: "https://creativecommons.org/publicdomain/zero/1.0/",
    # CC1
    LICENSE_CC_BY_1: "https://creativecommons.org/licenses/by/1.0/",
    LICENSE_CC_BY_SA_1: "https://creativecommons.org/licenses/by-sa/1.0/",
    LICENSE_CC_BY_ND_1: "https://creativecommons.org/licenses/by-nd/1.0/",
    LICENSE_CC_BY_NC_1: "https://creativecommons.org/licenses/by-nc/1.0/",
    LICENSE_CC_BY_NC_SA_1: "https://creativecommons.org/licenses/by-nc-sa/1.0/",
    LICENSE_CC_BY_ND_NC_1: "https://creativecommons.org/licenses/by-nc-nd/1.0/", # This year was reverse order from all subsequent years
    # CC2
    LICENSE_CC_BY_2: "https://creativecommons.org/licenses/by/2.0/",
    LICENSE_CC_BY_SA_2: "https://creativecommons.org/licenses/by-sa/2.0/",
    LICENSE_CC_BY_ND_2: "https://creativecommons.org/licenses/by-nd/2.0/",
    LICENSE_CC_BY_NC_2: "https://creativecommons.org/licenses/by-nc/2.0/",
    LICENSE_CC_BY_NC_SA_2: "https://creativecommons.org/licenses/by-nc-sa/2.0/",
    LICENSE_CC_BY_NC_ND_2: "https://creativecommons.org/licenses/by-nc-nd/2.0/",
    # CC2.5
    LICENSE_CC_BY_25: "https://creativecommons.org/licenses/by/2.5/",
    LICENSE_CC_BY_SA_25: "https://creativecommons.org/licenses/by-sa/2.5/",
    LICENSE_CC_BY_ND_25: "https://creativecommons.org/licenses/by-nd/2.5/",
    LICENSE_CC_BY_NC_25: "https://creativecommons.org/licenses/by-nc/2.5/",
    LICENSE_CC_BY_NC_SA_25: "https://creativecommons.org/licenses/by-nc-sa/2.5/",
    LICENSE_CC_BY_NC_ND_25: "https://creativecommons.org/licenses/by-nc-nd/2.5/",
    # CC3
    LICENSE_CC_BY_3: "https://creativecommons.org/licenses/by/3.0/",
    LICENSE_CC_BY_SA_3: "https://creativecommons.org/licenses/by-sa/3.0/",
    LICENSE_CC_BY_ND_3: "https://creativecommons.org/licenses/by-nd/3.0/",
    LICENSE_CC_BY_NC_3: "https://creativecommons.org/licenses/by-nc/3.0/",
    LICENSE_CC_BY_NC_SA_3: "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    LICENSE_CC_BY_NC_ND_3: "https://creativecommons.org/licenses/by-nc-nd/3.0/",
    # CC4
    LICENSE_CC_BY_4: "https://creativecommons.org/licenses/by/4.0/",
    LICENSE_CC_BY_SA_4: "https://creativecommons.org/licenses/by-sa/4.0/",
    LICENSE_CC_BY_ND_4: "https://creativecommons.org/licenses/by-nd/4.0/",
    LICENSE_CC_BY_NC_4: "https://creativecommons.org/licenses/by-nc/4.0/",
    LICENSE_CC_BY_NC_SA_4: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    LICENSE_CC_BY_NC_ND_4: "https://creativecommons.org/licenses/by-nc-nd/4.0/",
}
LICENSE_NAMES = {
    LICENSE_MIT: "MIT License",
    LICENSE_APACHE: "Apache License 2.0",
    LICENSE_MPL: "Mozilla Public License 2.0",
    LICENSE_OPENRAIL: "OpenRAIL-M License",
    LICENSE_OPENRAILPP: "OpenRAIL++-M License",
    # CC0
    LICENSE_CC0: "CC0 1.0 Universal",
    # CC1
    LICENSE_CC_BY_1: "CC BY 1.0",
    LICENSE_CC_BY_SA_1: "CC BY-SA 1.0",
    LICENSE_CC_BY_ND_1: "CC BY-ND 1.0",
    LICENSE_CC_BY_NC_1: "CC BY-NC 1.0",
    LICENSE_CC_BY_NC_SA_1: "CC BY-NC-SA 1.0",
    LICENSE_CC_BY_ND_NC_1: "CC BY-ND-NC 1.0", # Reverse order from all subsequent years
    # CC2
    LICENSE_CC_BY_2: "CC BY 2.0",
    LICENSE_CC_BY_SA_2: "CC BY-SA 2.0",
    LICENSE_CC_BY_ND_2: "CC BY-ND 2.0",
    LICENSE_CC_BY_NC_2: "CC BY-NC 2.0",
    LICENSE_CC_BY_NC_SA_2: "CC BY-NC-SA 2.0",
    LICENSE_CC_BY_NC_ND_2: "CC BY-NC-ND 2.0",
    # CC2.5
    LICENSE_CC_BY_25: "CC BY 2.5",
    LICENSE_CC_BY_SA_25: "CC BY-SA 2.5",
    LICENSE_CC_BY_ND_25: "CC BY-ND 2.5",
    LICENSE_CC_BY_NC_25: "CC BY-NC 2.5",
    LICENSE_CC_BY_NC_SA_25: "CC BY-NC-SA 2.5",
    LICENSE_CC_BY_NC_ND_25: "CC BY-NC-ND 2.5",
    # CC3
    LICENSE_CC_BY_3: "CC BY 3.0",
    LICENSE_CC_BY_SA_3: "CC BY-SA 3.0",
    LICENSE_CC_BY_ND_3: "CC BY-ND 3.0",
    LICENSE_CC_BY_NC_3: "CC BY-NC 3.0",
    LICENSE_CC_BY_NC_SA_3: "CC BY-NC-SA 3.0",
    LICENSE_CC_BY_NC_ND_3: "CC BY-NC-ND 3.0",
    # CC4
    LICENSE_CC_BY_4: "CC BY 4.0",
    LICENSE_CC_BY_SA_4: "CC BY-SA 4.0",
    LICENSE_CC_BY_ND_4: "CC BY-ND 4.0",
    LICENSE_CC_BY_NC_4: "CC BY-NC 4.0",
    LICENSE_CC_BY_NC_SA_4: "CC BY-NC-SA 4.0",
    LICENSE_CC_BY_NC_ND_4: "CC BY-NC-ND 4.0",
}
# (attribution required, derivatives allowed, redistribution allowed, copyleft required, commercial allowed, hosting allowed)
LICENSE_ALLOWANCES = {
    LICENSE_MIT: (True, True, True, False, True, True),
    LICENSE_APACHE: (True, True, True, False, True, True),
    LICENSE_MPL: (True, True, True, True, True, True),
    LICENSE_OPENRAIL: (True, True, True, True, True, True),
    LICENSE_OPENRAILPP: (True, True, True, True, True, True),
    # CC0
    LICENSE_CC0: (False, True, True, False, True, True),
    # CC1
    LICENSE_CC_BY_1: (True, True, True, False, True, True),
    LICENSE_CC_BY_SA_1: (True, True, True, True, True, True),
    LICENSE_CC_BY_ND_1: (True, False, True, False, True, True),
    LICENSE_CC_BY_NC_1: (True, True, True, False, False, True),
    LICENSE_CC_BY_NC_SA_1: (True, True, True, True, False, True),
    LICENSE_CC_BY_ND_NC_1: (True, False, True, False, False, True),
    # CC2
    LICENSE_CC_BY_2: (True, True, True, False, True, True),
    LICENSE_CC_BY_SA_2: (True, True, True, True, True, True),
    LICENSE_CC_BY_ND_2: (True, False, True, False, True, True),
    LICENSE_CC_BY_NC_2: (True, True, True, False, False, True),
    LICENSE_CC_BY_NC_SA_2: (True, True, True, True, False, True),
    LICENSE_CC_BY_NC_ND_2: (True, False, True, False, False, True),
    # CC25
    LICENSE_CC_BY_25: (True, True, True, False, True, True),
    LICENSE_CC_BY_SA_25: (True, True, True, True, True, True),
    LICENSE_CC_BY_ND_25: (True, False, True, False, True, True),
    LICENSE_CC_BY_NC_25: (True, True, True, False, False, True),
    LICENSE_CC_BY_NC_SA_25: (True, True, True, True, False, True),
    LICENSE_CC_BY_NC_ND_25: (True, False, True, False, False, True),
    # CC3
    LICENSE_CC_BY_3: (True, True, True, False, True, True),
    LICENSE_CC_BY_SA_3: (True, True, True, True, True, True),
    LICENSE_CC_BY_ND_3: (True, False, True, False, True, True),
    LICENSE_CC_BY_NC_3: (True, True, True, False, False, True),
    LICENSE_CC_BY_NC_SA_3: (True, True, True, True, False, True),
    LICENSE_CC_BY_NC_ND_3: (True, False, True, False, False, True),
    # CC4
    LICENSE_CC_BY_4: (True, True, True, False, True, True),
    LICENSE_CC_BY_SA_4: (True, True, True, True, True, True),
    LICENSE_CC_BY_ND_4: (True, False, True, False, True, True),
    LICENSE_CC_BY_NC_4: (True, True, True, False, False, True),
    LICENSE_CC_BY_NC_SA_4: (True, True, True, True, False, True),
    LICENSE_CC_BY_NC_ND_4: (True, False, True, False, False, True),
}
