# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import math
from libs.Utils import Utils

#All FINAL  (for Training)
# (bad subjects removed: 994273, 937160, 885975, 788876, 713239)
# (no CA: 885975, 788876, 713239)
all_subjects_FINAL = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
                    "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "907656",
                    "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
                    "887373", "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
                    "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
                    "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
                    "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
                    "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
                    "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
                    "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
                    "620434", "613538", "601127", "599671", "599469"]   #105

#All With Outliers  (for Preprocessing)
all_subjects_RAW = ["994273", "992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574", "958976",
                "957974", "951457", "937160", "932554", "930449", "922854", "917255", "912447", "910241", "907656", "904044", "901442",
                "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "887373", "885975", "877269", "877168",
                "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766", "849971", "845458", "837964",
                "837560", "833249", "833148", "826454", "826353", "816653", "814649", "802844", "792766", "792564", "789373", "788876",
                "786569", "784565", "782561", "779370", "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348",
                "749361", "748662", "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "713239", "709551",
                "705341", "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455",
                "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538", "601127",
                "599671", "599469"] #110

def get_all_subjects():
    '''
    This can be imported in other parts of project to get subjects
    '''
    return all_subjects_FINAL

def get_all_subjects_RAW():
    return all_subjects_RAW

def get_subjects_chunk(nr_batches, batch_number):
    nr_batches = int(nr_batches)
    batch_number = int(batch_number)

    batch_size = int(math.ceil(len(all_subjects_RAW) / float(nr_batches)))
    res = list(Utils.chunks(all_subjects_RAW, batch_size))
    final_subjects = res[batch_number]
    return final_subjects

def main():
    '''
    This can be used in Shell scripts to get subjects
    '''
    args = sys.argv[1:]
    nr_batches = int(args[0])  # Number of batches
    batch_number = int(args[1])  # Which batch do we want     (idx starts at 0)

    batch_size = int(math.ceil(len(all_subjects_RAW) / float(nr_batches)))
    res = list(Utils.chunks(all_subjects_RAW, batch_size))

    #Note: can not print anyhting, because goes as parameter to script
    # print("Nr of Batches: {} (last batch might be smaller)".format(len(res)))
    # print("Nr of subjects in batch: {}".format(batch_size))
    final_subjects = res[batch_number]
    # print("Subjects: {}".format(final_subjects))

    #To String:
    str = ""
    for subject in final_subjects:
        str += subject + " "
    str = str[:-1]  #remove last space
    print(str)

if __name__ == "__main__":
    main()