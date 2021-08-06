# Universal Classification Files
---

This folder contains the Universal Classification Files developed in [Ma (2021, 668)](https://doi.org/10.1177/0899764020968153). It primarily collects NTEE codes from the BMF file and texts from the IRS 990 Forms. The paper introduces more details.

# Download files in other formats:
---

- MS Excel format:
    - Train: https://osf.io/569dn/
    - Test: https://osf.io/es5gf/
- Plain text format (tab seperated):
    - Train: https://osf.io/qk4be/
    - Test: https://osf.io/7ms9n/

# Codebook
---

|Letter code | Meaning | Source|
|---|---|---|
|DLN|[Document Locator Number](https://web.archive.org/web/20170519181012/https://www.irs.gov/pub/irs-utl/6209-section4-2012.pdf)| AWS-Index|
|EIN|Employer Identification Number|AWS-Index|
|FILING_TYPE|Filing type|AWS-Index|
|IRS990EZ_p3_DscrptnPrgrmSrvcAccmTxt| 990-EZ Part III Lines 28-30, program description|AWS-XML|
|IRS990EZ_p3_PrmryExmptPrpsTxt| 990-EZ Part III Lines 28-30, program description|AWS-XML|
|IRS990PF_p16b_RltnshpSttmntTxt| 990-PF Part XVI-B, program description|AWS-XML|
|IRS990PF_p9a_DscrptnTxt| 990-PF Part IX-A, program description|AWS-XML|
|IRS990ScheduleO_ExplntnTxt| Schedule O, program description|AWS-XML|
|IRS990_p1_ActvtyOrMssnDsc| 990 Part I Line 1, mission statement|AWS-XML|
|IRS990_p3_DscS|990 Part III Line 4, program description|AWS-XML|
|IRS990_p3_MssnDsc|990 Part III Line 1, mission statement|AWS-XML|
|OBJECT_ID| Object ID (XML file unique identifier)|AWS-Index|
|RETURN_ID| Return ID|AWS-Index|
|RETURN_TYPE|Return type|AWS-Index|
|SUB_DATE|Submission date|AWS-Index|
|TAXPAYER_NAME|Taxpayer name (i.e., organization name)|AWS-Index|
|TAX_PERIOD|Tax period|AWS-Index|
|YEAR|Tax year|AWS-Index|
|95_and_before|Founded in 1995 and before (1=yes, 0=no)|Self-coded|
|NTEE1|NTEE major group|BMF|
|mission|All mission statement texts, duplicates removed|Self-coded|
|prgrm_dsc|All program description texts, duplicates removed|Self-coded|
|mission_spellchk|`mission` column spell-checked|Self-coded|
|prgrm_dsc_spellchk|`prgrm_dsc` column spell-checked|Self-coded|

Note:

- AWS: [IRS 990 Filings on AWS](https://registry.opendata.aws/irs990/)
    - Index: Index file, for example: https://s3.amazonaws.com/irs-form-990/index_2011.csv
    - XML: Electronic 990 filing, for example: https://s3.amazonaws.com/irs-form-990/201541349349307794_public.xml
- BMF: [Exempt Organizations Business Master File Extract](https://www.irs.gov/charities-non-profits/exempt-organizations-business-master-file-extract-eo-bmf)
