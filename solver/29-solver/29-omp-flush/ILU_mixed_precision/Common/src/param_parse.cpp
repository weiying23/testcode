
/*  A Bison parser, made from param_file_parse.y with Bison version GNU Bison version 1.24
  */
/* yyerror abort together */
#define YYBISON 1  /* Identify Bison output.  */

#define CONST   258
#define TYPEINT 259
#define TYPEREAL    260
#define TYPEDOUBL   261
#define TYPEFLOAT   262
#define TYPESTR 263
#define VARNAME 264
#define INTEGER 265
#define REALNUM 266
#define STRING  267
#define ACTSCOPE    268
#define UMINUS  269

    #include <string>
    #include <vector>
    
    #include "parameters.h"

    typedef struct
    {
        union _val{
            bool b;
            int i;
            real r;
            double d;
            float f;
        }val;
        
        std::string s;
        
        std::vector<int> v_i;
        std::vector<real> v_r;
        std::vector<double> v_d;
        std::vector<float> v_f;
        std::vector<std::string> v_s;
        
        std::vector< std::vector<int> > vv_i;
        std::vector< std::vector<real> > vv_r;
        std::vector< std::vector<double> > vv_d;
        std::vector< std::vector<float> > vv_f;
        std::vector< std::vector<std::string> > vv_s;
    }YYSTYPE;
    
    extern YYSTYPE yylval;

    #include "param_lexical.h"

    //extern FILE * yyin;
    //extern int line_no;
    
    parameter_space::parameters * global_params = _nullptr;

    std::vector<std::string> nested_group_names;
    
    //int read_parameters( const std::string & file_name, parameter_space::parameters & params );
    void yyerror( const std::string & msg );
    extern int yylex();

#ifndef YYLTYPE
typedef
  struct yyltype
    {
      int timestamp;
      int first_line;
      int first_column;
      int last_line;
      int last_column;
      char *text;
   }
  yyltype;

#define YYLTYPE yyltype
#endif

#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define YYFINAL     250
#define YYFLAG      -32768
#define YYNTBASE    31

#define YYTRANSLATE(x) ((unsigned)(x) <= 269 ? yytranslate[x] : 63)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,    18,     2,     2,    25,
    26,    16,    14,    27,    15,     2,    17,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    28,    21,     2,
    23,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    29,     2,    30,    20,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    22,     2,    24,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     1,     2,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    19
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     1,     5,     8,    11,    15,    19,    25,    31,    37,
    43,    49,    56,    63,    70,    77,    84,    94,   104,   114,
   124,   134,   135,   137,   139,   141,   145,   149,   153,   157,
   161,   164,   167,   171,   175,   177,   179,   183,   187,   191,
   195,   198,   201,   205,   207,   209,   213,   217,   221,   225,
   228,   231,   235,   237,   239,   243,   247,   251,   255,   258,
   261,   265,   267,   269,   271,   275,   279,   283,   287,   289,
   292,   295,   299,   301,   303,   307,   311,   314,   318,   321,
   325,   327,   331,   334,   337,   341,   343,   347,   350,   353,
   357,   359,   363,   366,   369,   373,   375,   377,   381,   385,
   388,   391,   395,   397,   399,   403,   407,   411,   415,   419,
   422,   425,   429,   433,   435,   439,   440,   446,   447,   452,
   454,   458,   461,   463,   467,   470,   472,   476,   479,   481,
   485,   488,   490,   494
};

static const short yyrhs[] = {    -1,
    31,    32,    21,     0,    31,    55,     0,    31,    21,     0,
    31,     1,    21,     0,    31,     1,    22,     0,    33,     4,
     9,    23,    34,     0,    33,     5,     9,    23,    35,     0,
    33,     6,     9,    23,    36,     0,    33,     7,     9,    23,
    37,     0,    33,     8,     9,    23,    38,     0,    33,     4,
     9,    52,    23,    41,     0,    33,     5,     9,    52,    23,
    44,     0,    33,     6,     9,    52,    23,    46,     0,    33,
     7,     9,    52,    23,    48,     0,    33,     8,     9,    52,
    23,    50,     0,    33,     4,     9,    52,    52,    23,    22,
    58,    24,     0,    33,     5,     9,    52,    52,    23,    22,
    59,    24,     0,    33,     6,     9,    52,    52,    23,    22,
    60,    24,     0,    33,     7,     9,    52,    52,    23,    22,
    61,    24,     0,    33,     8,     9,    52,    52,    23,    22,
    62,    24,     0,     0,     3,     0,    10,     0,    54,     0,
    34,    14,    34,     0,    34,    15,    34,     0,    34,    16,
    34,     0,    34,    17,    34,     0,    34,    18,    34,     0,
    14,    34,     0,    15,    34,     0,    34,    20,    34,     0,
    25,    34,    26,     0,    11,     0,    54,     0,    35,    14,
    35,     0,    35,    15,    35,     0,    35,    16,    35,     0,
    35,    17,    35,     0,    14,    35,     0,    15,    35,     0,
    25,    35,    26,     0,    11,     0,    54,     0,    36,    14,
    36,     0,    36,    15,    36,     0,    36,    16,    36,     0,
    36,    17,    36,     0,    14,    36,     0,    15,    36,     0,
    25,    36,    26,     0,    11,     0,    54,     0,    37,    14,
    37,     0,    37,    15,    37,     0,    37,    16,    37,     0,
    37,    17,    37,     0,    14,    37,     0,    15,    37,     0,
    25,    37,    26,     0,    40,     0,    54,     0,    39,     0,
    40,    14,    54,     0,    54,    14,    40,     0,    39,    14,
    40,     0,    39,    14,    54,     0,    12,     0,    40,    12,
     0,    22,    24,     0,    22,    42,    24,     0,    34,     0,
    43,     0,    42,    27,    34,     0,    42,    27,    43,     0,
    42,    27,     0,    10,    28,    10,     0,    22,    24,     0,
    22,    45,    24,     0,    35,     0,    45,    27,    35,     0,
    45,    27,     0,    22,    24,     0,    22,    47,    24,     0,
    36,     0,    47,    27,    36,     0,    47,    27,     0,    22,
    24,     0,    22,    49,    24,     0,    37,     0,    49,    27,
    37,     0,    49,    27,     0,    22,    24,     0,    22,    51,
    24,     0,    12,     0,    54,     0,    51,    27,    12,     0,
    51,    27,    54,     0,    51,    27,     0,    29,    30,     0,
    29,    53,    30,     0,    10,     0,    54,     0,    53,    14,
    53,     0,    53,    15,    53,     0,    53,    16,    53,     0,
    53,    17,    53,     0,    53,    18,    53,     0,    14,    53,
     0,    15,    53,     0,    53,    20,    53,     0,    25,    53,
    26,     0,     9,     0,    54,    13,     9,     0,     0,     9,
    22,    56,    31,    24,     0,     0,    22,    57,    31,    24,
     0,    41,     0,    58,    27,    41,     0,    58,    27,     0,
    44,     0,    59,    27,    44,     0,    59,    27,     0,    46,
     0,    60,    27,    46,     0,    60,    27,     0,    48,     0,
    61,    27,    48,     0,    61,    27,     0,    50,     0,    62,
    27,    50,     0,    62,    27,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    77,    78,    79,    80,    81,    82,    85,    86,    87,    88,
    89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
    99,   102,   103,   106,   107,   108,   109,   110,   111,   112,
   113,   114,   115,   116,   119,   120,   121,   122,   123,   124,
   125,   126,   127,   130,   131,   132,   133,   134,   135,   136,
   137,   138,   141,   142,   143,   144,   145,   146,   147,   148,
   149,   152,   153,   154,   157,   158,   159,   160,   163,   164,
   167,   168,   171,   172,   173,   174,   175,   178,   181,   182,
   185,   186,   187,   190,   191,   194,   195,   196,   199,   200,
   203,   204,   205,   208,   209,   212,   213,   215,   216,   217,
   220,   221,   224,   225,   228,   229,   230,   231,   232,   233,
   234,   235,   236,   239,   240,   243,   243,   244,   245,   248,
   249,   250,   253,   254,   255,   258,   259,   260,   263,   264,
   265,   268,   269,   270
};

static const char * const yytname[] = {   "$","error","$undefined.","CONST",
"TYPEINT","TYPEREAL","TYPEDOUBL","TYPEFLOAT","TYPESTR","VARNAME","INTEGER","REALNUM",
"STRING","ACTSCOPE","'+'","'-'","'*'","'/'","'%'","UMINUS","'^'","';'","'{'",
"'='","'}'","'('","')'","','","':'","'['","']'","lines","line","isconst","exprint",
"expreal","exprdbl","exprflt","exprstr","plusstr","catstrs","blankidsints","idsints",
"rangeints","blankidsreals","idsreals","blankidsdoubls","idsdoubles","blankidsfloats",
"idsfloats","blankidsstrs","idsstrs","arraysize","constexprint","scopvar","groupproc",
"@1","@2","dblblankidsints","dblblankidsreals","dblblankidsdoubls","dblblankidsfloats",
"dblblankidsstrs",""
};
#endif

static const short yyr1[] = {     0,
    31,    31,    31,    31,    31,    31,    32,    32,    32,    32,
    32,    32,    32,    32,    32,    32,    32,    32,    32,    32,
    32,    33,    33,    34,    34,    34,    34,    34,    34,    34,
    34,    34,    34,    34,    35,    35,    35,    35,    35,    35,
    35,    35,    35,    36,    36,    36,    36,    36,    36,    36,
    36,    36,    37,    37,    37,    37,    37,    37,    37,    37,
    37,    38,    38,    38,    39,    39,    39,    39,    40,    40,
    41,    41,    42,    42,    42,    42,    42,    43,    44,    44,
    45,    45,    45,    46,    46,    47,    47,    47,    48,    48,
    49,    49,    49,    50,    50,    51,    51,    51,    51,    51,
    52,    52,    53,    53,    53,    53,    53,    53,    53,    53,
    53,    53,    53,    54,    54,    56,    55,    57,    55,    58,
    58,    58,    59,    59,    59,    60,    60,    60,    61,    61,
    61,    62,    62,    62
};

static const short yyr2[] = {     0,
     0,     3,     2,     2,     3,     3,     5,     5,     5,     5,
     5,     6,     6,     6,     6,     6,     9,     9,     9,     9,
     9,     0,     1,     1,     1,     3,     3,     3,     3,     3,
     2,     2,     3,     3,     1,     1,     3,     3,     3,     3,
     2,     2,     3,     1,     1,     3,     3,     3,     3,     2,
     2,     3,     1,     1,     3,     3,     3,     3,     2,     2,
     3,     1,     1,     1,     3,     3,     3,     3,     1,     2,
     2,     3,     1,     1,     3,     3,     2,     3,     2,     3,
     1,     3,     2,     2,     3,     1,     3,     2,     2,     3,
     1,     3,     2,     2,     3,     1,     1,     3,     3,     2,
     2,     3,     1,     1,     3,     3,     3,     3,     3,     2,
     2,     3,     3,     1,     3,     0,     5,     0,     4,     1,
     3,     2,     1,     3,     2,     1,     3,     2,     1,     3,
     2,     1,     3,     2
};

static const short yydefact[] = {     1,
     0,     0,    23,     0,     4,   118,     0,     0,     3,     5,
     6,   116,     1,     2,     0,     0,     0,     0,     0,     1,
     0,     0,     0,     0,     0,     0,     0,   119,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   117,
   114,    24,     0,     0,     0,     7,    25,   103,     0,     0,
     0,   101,     0,   104,     0,     0,    35,     0,     0,     0,
     8,    36,     0,     0,    44,     0,     0,     0,     9,    45,
     0,     0,    53,     0,     0,     0,    10,    54,     0,     0,
    69,    11,    64,    62,    63,     0,     0,    31,    32,     0,
     0,     0,     0,     0,     0,     0,     0,   110,   111,     0,
     0,     0,     0,     0,     0,     0,   102,     0,    12,     0,
    41,    42,     0,     0,     0,     0,     0,     0,    13,     0,
    50,    51,     0,     0,     0,     0,     0,     0,    14,     0,
    59,    60,     0,     0,     0,     0,     0,     0,    15,     0,
     0,    70,     0,     0,     0,    16,     0,    34,    26,    27,
    28,    29,    30,    33,   115,   113,   105,   106,   107,   108,
   109,   112,    24,    71,    73,     0,    74,     0,    43,    37,
    38,    39,    40,    79,    81,     0,     0,    52,    46,    47,
    48,    49,    84,    86,     0,     0,    61,    55,    56,    57,
    58,    89,    91,     0,     0,    67,    68,    65,    66,    96,
    94,     0,    97,     0,     0,    72,    77,   120,     0,    80,
    83,   123,     0,    85,    88,   126,     0,    90,    93,   129,
     0,    95,   100,   132,     0,    78,    75,    76,    17,   122,
    82,    18,   125,    87,    19,   128,    92,    20,   131,    98,
    99,    21,   134,   121,   124,   127,   130,   133,     0,     0
};

static const short yydefgoto[] = {     1,
     7,     8,    46,    61,    69,    77,    82,    83,    84,   109,
   166,   167,   119,   176,   129,   185,   139,   194,   146,   202,
    31,    53,    47,     9,    20,    13,   209,   213,   217,   221,
   225
};

static const short yypact[] = {-32768,
   191,     4,-32768,   -15,-32768,-32768,   -10,   315,-32768,-32768,
-32768,-32768,-32768,-32768,    38,    59,    61,    74,    93,-32768,
   120,    80,   181,   182,   194,   204,   162,-32768,   229,    25,
   207,   236,   223,   244,   234,   251,   241,    57,   248,-32768,
-32768,-32768,   229,   229,   229,   298,    97,-32768,   258,   258,
   258,-32768,   116,    97,    90,    91,-32768,   236,   236,   236,
   316,    97,   100,   124,-32768,   244,   244,   244,   320,    97,
   117,   151,-32768,   251,   251,   251,   324,    97,   135,   167,
-32768,-32768,   201,    40,    29,   260,   206,   264,   264,   271,
   229,   229,   229,   229,   229,   229,   247,   283,   283,   278,
   258,   258,   258,   258,   258,   258,-32768,   163,-32768,   305,
-32768,-32768,     1,   236,   236,   236,   236,   134,-32768,   307,
-32768,-32768,   285,   244,   244,   244,   244,   210,-32768,   322,
-32768,-32768,   291,   251,   251,   251,   251,   217,-32768,   326,
    57,-32768,   350,   349,   126,-32768,   340,-32768,   144,   144,
   264,   264,   264,   264,-32768,-32768,   308,   308,   283,   283,
   283,   283,   335,-32768,   298,   113,-32768,    90,-32768,   186,
   186,-32768,-32768,-32768,   316,   155,   100,-32768,   232,   232,
-32768,-32768,-32768,   320,   213,   117,-32768,   293,   293,-32768,
-32768,-32768,   324,   254,   135,   352,    97,    97,   352,-32768,
-32768,   318,    97,   260,   355,-32768,   265,-32768,   319,-32768,
   236,-32768,   323,-32768,   244,-32768,   325,-32768,   251,-32768,
   327,-32768,   344,-32768,   331,-32768,   298,-32768,-32768,    90,
   316,-32768,   100,   320,-32768,   117,   324,-32768,   135,-32768,
    97,-32768,   260,-32768,-32768,-32768,-32768,-32768,   366,-32768
};

static const short yypgoto[] = {    28,
-32768,-32768,   -31,   -36,   -35,   -18,-32768,-32768,   216,  -163,
-32768,   160,  -174,-32768,  -185,-32768,  -186,-32768,  -194,-32768,
   183,    50,   -30,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768
};


#define YYLAST      367


static const short yytable[] = {    54,
   216,    62,   212,    70,   208,    78,    12,    85,   220,   224,
    14,    88,    89,    90,   114,   115,   116,   117,    54,    54,
    54,   111,   112,   113,    10,    11,   169,    62,    62,    62,
   121,   122,   123,    41,    48,    70,    70,    70,    49,    50,
    21,    97,   144,    78,    78,    78,    22,    27,   248,    51,
   246,   142,   247,   143,    52,   131,   132,   133,   245,   149,
   150,   151,   152,   153,   154,    41,   244,    23,    81,    24,
    54,    54,    54,    54,    54,    54,   165,   170,   171,   172,
   173,   175,    25,    62,    62,    62,    62,    62,   179,   180,
   181,   182,   184,    70,    70,    70,    70,    70,    98,    99,
   100,    26,    29,    78,    78,    78,    78,    78,    30,    97,
   197,   108,   198,   110,   203,   188,   189,   190,   191,   193,
     2,   118,     3,   -22,   -22,   -22,   -22,   -22,     4,   101,
   102,   103,   104,   105,    41,   106,   206,   200,   128,   207,
     5,     6,    41,    28,    57,   107,   120,    58,    59,   201,
   157,   158,   159,   160,   161,   162,   138,   174,    60,    93,
    94,    95,     2,    96,     3,   -22,   -22,   -22,   -22,   -22,
     4,    41,   163,   130,   231,   227,    43,    44,   210,   234,
    62,   211,     5,     6,    70,    40,   164,    45,    78,   140,
   249,     2,   241,     3,   -22,   -22,   -22,   -22,   -22,     4,
   237,   116,   117,    32,    34,    33,    35,    37,    39,    30,
    30,     5,     6,    56,   141,    64,    36,    72,    41,    80,
    65,    87,    30,    66,    67,    41,    38,    73,   147,    55,
    74,    75,    30,   183,    68,    30,   214,    41,    42,   215,
   192,    76,    43,    44,    41,    63,    57,   126,   127,    58,
    59,    30,    41,    45,    65,   155,    71,    66,    67,    41,
    60,    73,    30,    79,    74,    75,    41,    48,    68,    30,
    86,    49,    50,    41,   163,    76,    30,   218,    43,    44,
   219,   145,    51,    96,    91,    92,    93,    94,    95,    45,
    96,   101,   102,   103,   104,   105,   148,   106,   124,   125,
   126,   127,   106,   156,   134,   135,   136,   137,   136,   137,
   178,    91,    92,    93,    94,    95,   187,    96,    15,    16,
    17,    18,    19,   103,   104,   105,   168,   106,   177,   114,
   115,   116,   117,   124,   125,   126,   127,   134,   135,   136,
   137,   222,   229,   186,   223,   230,   232,   195,   235,   233,
   238,   236,    41,   239,   242,   240,   196,   243,    41,   199,
    81,   204,   205,   142,   226,   250,   228
};

static const short yycheck[] = {    30,
   186,    32,   177,    34,   168,    36,    22,    38,   195,   204,
    21,    43,    44,    45,    14,    15,    16,    17,    49,    50,
    51,    58,    59,    60,    21,    22,    26,    58,    59,    60,
    66,    67,    68,     9,    10,    66,    67,    68,    14,    15,
    13,    13,    14,    74,    75,    76,     9,    20,   243,    25,
   236,    12,   239,    14,    30,    74,    75,    76,   233,    91,
    92,    93,    94,    95,    96,     9,   230,     9,    12,     9,
   101,   102,   103,   104,   105,   106,   108,   114,   115,   116,
   117,   118,     9,   114,   115,   116,   117,   118,   124,   125,
   126,   127,   128,   124,   125,   126,   127,   128,    49,    50,
    51,     9,    23,   134,   135,   136,   137,   138,    29,    13,
   141,    22,   143,    23,   145,   134,   135,   136,   137,   138,
     1,    22,     3,     4,     5,     6,     7,     8,     9,    14,
    15,    16,    17,    18,     9,    20,    24,    12,    22,    27,
    21,    22,     9,    24,    11,    30,    23,    14,    15,    24,
   101,   102,   103,   104,   105,   106,    22,    24,    25,    16,
    17,    18,     1,    20,     3,     4,     5,     6,     7,     8,
     9,     9,    10,    23,   211,   207,    14,    15,    24,   215,
   211,    27,    21,    22,   215,    24,    24,    25,   219,    23,
     0,     1,   223,     3,     4,     5,     6,     7,     8,     9,
   219,    16,    17,    23,    23,    23,    24,    25,    26,    29,
    29,    21,    22,    31,    14,    33,    23,    35,     9,    37,
    11,    39,    29,    14,    15,     9,    23,    11,    23,    23,
    14,    15,    29,    24,    25,    29,    24,     9,    10,    27,
    24,    25,    14,    15,     9,    23,    11,    16,    17,    14,
    15,    29,     9,    25,    11,     9,    23,    14,    15,     9,
    25,    11,    29,    23,    14,    15,     9,    10,    25,    29,
    23,    14,    15,     9,    10,    25,    29,    24,    14,    15,
    27,    22,    25,    20,    14,    15,    16,    17,    18,    25,
    20,    14,    15,    16,    17,    18,    26,    20,    14,    15,
    16,    17,    20,    26,    14,    15,    16,    17,    16,    17,
    26,    14,    15,    16,    17,    18,    26,    20,     4,     5,
     6,     7,     8,    16,    17,    18,    22,    20,    22,    14,
    15,    16,    17,    14,    15,    16,    17,    14,    15,    16,
    17,    24,    24,    22,    27,    27,    24,    22,    24,    27,
    24,    27,     9,    27,    24,    12,   141,    27,     9,   144,
    12,    22,    28,    12,    10,     0,   207
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
// 3 "bison.simple"

/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

#ifdef __GNUC__
#ifndef alloca
#define alloca __builtin_alloca
#endif
#else /* not __GNUC__ */
#if HAVE_ALLOCA_H
#include <alloca.h>
#else /* not HAVE_ALLOCA_H */
#ifdef _AIX
 #pragma alloca
#else /* not _AIX */
char *alloca ();
#endif /* not _AIX */
#endif /* not HAVE_ALLOCA_H */
#endif /* not __GNUC__ */

extern int yylex();
extern void yyerror();

#ifndef alloca
#ifdef __GNUC__
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi)
#include <alloca.h>
#else /* not sparc */
#if defined (MSDOS) && !defined (__TURBOC__)
#include <malloc.h>
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
#include <malloc.h>
 #pragma alloca
#else /* not MSDOS, __TURBOC__, or _AIX */
#ifdef __hpux
#ifdef __cplusplus
extern "C" {
void *alloca (unsigned int);
};
#else /* not __cplusplus */
void *alloca ();
#endif /* not __cplusplus */
#endif /* __hpux */
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc.  */
#endif /* not GNU C.  */
#endif /* alloca not defined.  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok     (yyerrstatus = 0)
#define yyclearin   (yychar = YYEMPTY)
#define YYEMPTY     -2
#define YYEOF       0
#define YYACCEPT    return(0)
#define YYABORT     return(1)
#define YYERROR     goto yyerrlab1
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL      goto yyerrlab
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do                              \
  if (yychar == YYEMPTY && yylen == 1)              \
    { yychar = (token), yylval = (value);           \
      yychar1 = YYTRANSLATE (yychar);               \
      YYPOPSTACK;                       \
      goto yybackup;                        \
    }                               \
  else                              \
    { yyerror ("syntax error: cannot back up"); YYERROR; }  \
while (0)

#define YYTERROR    1
#define YYERRCODE   256

#ifndef YYPURE
#define YYLEX       yylex()
#endif

#ifdef YYPURE
#ifdef YYLSP_NEEDED
#ifdef YYLEX_PARAM
#define YYLEX       yylex(&yylval, &yylloc, YYLEX_PARAM)
#else
#define YYLEX       yylex(&yylval, &yylloc)
#endif
#else /* not YYLSP_NEEDED */
#ifdef YYLEX_PARAM
#define YYLEX       yylex(&yylval, YYLEX_PARAM)
#else
#define YYLEX       yylex(&yylval)
#endif
#endif /* not YYLSP_NEEDED */
#endif

/* If nonreentrant, generate the variables here */

#ifndef YYPURE

int yychar;         /*  the lookahead symbol        */
YYSTYPE yylval;         /*  the semantic value of the       */
                /*  lookahead symbol            */

#ifdef YYLSP_NEEDED
YYLTYPE yylloc;         /*  location data for the lookahead */
                /*  symbol              */
#endif

int yynerrs;            /*  number of parse errors so far       */
#endif  /* not YYPURE */

#if YYDEBUG != 0
int yydebug;            /*  nonzero means print parse trace */
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks   */

#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
int yyparse (void);
#endif

#if __GNUC__ > 1        /* GNU C and GNU C++ define this.  */
#define __yy_memcpy(FROM,TO,COUNT)  __builtin_memcpy(TO,FROM,COUNT)
#else               /* not GNU C or C++ */
#ifndef __cplusplus

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (from, to, count)
     char *from;
     char *to;
     int count;
{
  char *f = from;
  char *t = to;
  int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#else /* __cplusplus */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (char *from, char *to, int count)
{
  char *f = from;
  char *t = to;
  int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

// 192 "bison.simple"

/* The user can define YYPARSE_PARAM as the name of an argument to be passed
   into yyparse.  The argument should have type void *.
   It should actually point to an object.
   Grammar actions can access the variable by casting it
   to the proper pointer type.  */

#ifdef YYPARSE_PARAM
#define YYPARSE_PARAM_DECL void *YYPARSE_PARAM;
#else
#define YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#endif

int
yyparse(YYPARSE_PARAM)
     YYPARSE_PARAM_DECL
{
  int yystate;
  int yyn;
  short *yyssp;
  YYSTYPE *yyvsp;
  int yyerrstatus;  /*  number of tokens to shift before error messages enabled */
  int yychar1 = 0;      /*  lookahead token as an internal (translated) token number */

  short yyssa[YYINITDEPTH]; /*  the state stack         */
  YYSTYPE yyvsa[YYINITDEPTH];   /*  the semantic value stack        */

  short *yyss = yyssa;      /*  refer to the stacks thru separate pointers */
  YYSTYPE *yyvs = yyvsa;    /*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YYLSP_NEEDED
  YYLTYPE yylsa[YYINITDEPTH];   /*  the location stack          */
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;

#ifdef YYPURE
  int yychar;
  YYSTYPE yylval;
  int yynerrs;
#ifdef YYLSP_NEEDED
  YYLTYPE yylloc;
#endif
#endif

  YYSTYPE yyval;        /*  the variable used to return     */
                /*  semantic values from the action */
                /*  routines                */

  int yylen;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Starting parse\n");
#endif

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;     /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YYLSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
yynewstate:

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YYSTYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YYLSP_NEEDED
      YYLTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
     the data in use in that stack, in bytes.  */
#ifdef YYLSP_NEEDED
      /* This used to be a conditional around just the two extra args,
     but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
         &yyss1, size * sizeof (*yyssp),
         &yyvs1, size * sizeof (*yyvsp),
         &yyls1, size * sizeof (*yylsp),
         &yystacksize);
#else
      yyoverflow("parser stack overflow",
         &yyss1, size * sizeof (*yyssp),
         &yyvs1, size * sizeof (*yyvsp),
         &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YYLSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
    {
      yyerror("parser stack overflow");
      return 2;
    }
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
    yystacksize = YYMAXDEPTH;
      yyss = (short *) alloca (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss1, (char *)yyss, size * sizeof (*yyssp));
      yyvs = (YYSTYPE *) alloca (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs1, (char *)yyvs, size * sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) alloca (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls1, (char *)yyls, size * sizeof (*yylsp));
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YYLSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YYDEBUG != 0
      if (yydebug)
    fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
    YYABORT;
    }

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  goto yybackup;
 yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (yychar == YYEMPTY)
    {
#if YYDEBUG != 0
      if (yydebug)
    fprintf(stderr, "Reading a token: ");
#endif
      yychar = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (yychar <= 0)      /* This means end of input. */
    {
      yychar1 = 0;
      yychar = YYEOF;       /* Don't call YYLEX any more */

#if YYDEBUG != 0
      if (yydebug)
    fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(yychar);

#if YYDEBUG != 0
      if (yydebug)
    {
      fprintf (stderr, "Next token is %d (%s", yychar, yytname[yychar1]);
      /* Give the individual parser a way to print the precise meaning
         of a token, for further debugging info.  */
#ifdef YYPRINT
      YYPRINT (stderr, yychar, yylval);
#endif
      fprintf (stderr, ")\n");
    }
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    goto yydefault;

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
    goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrlab;

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting token %d (%s), ", yychar, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  goto yynewstate;

/* Do the default action for the current state.  */
yydefault:

  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
yyreduce:
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YYDEBUG != 0
  if (yydebug)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
           yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
    fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


  switch (yyn) {

case 4:

{
    break;}
case 5:

{  yyerrok;
    break;}
case 6:

{  yyerrok;
    break;}
case 7:

{ global_params->add(yyvsp[-2].s, yyvsp[0].val.i, nested_group_names, yyvsp[-4].val.b); ;
    break;}
case 8:

{ global_params->add(yyvsp[-2].s, yyvsp[0].val.r, nested_group_names, yyvsp[-4].val.b); ;
    break;}
case 9:

{ global_params->add(yyvsp[-2].s, yyvsp[0].val.d, nested_group_names, yyvsp[-4].val.b); ;
    break;}
case 10:

{ global_params->add(yyvsp[-2].s, yyvsp[0].val.f, nested_group_names, yyvsp[-4].val.b); ;
    break;}
case 11:

{ global_params->add(yyvsp[-2].s, yyvsp[0].s, nested_group_names, yyvsp[-4].val.b); ;
    break;}
case 12:

{ global_params->add(yyvsp[-3].s, yyvsp[-2].val.i, yyvsp[0].v_i, nested_group_names, yyvsp[-5].val.b); ;
    break;}
case 13:

{ global_params->add(yyvsp[-3].s, yyvsp[-2].val.i, yyvsp[0].v_r, nested_group_names, yyvsp[-5].val.b); ;
    break;}
case 14:

{ global_params->add(yyvsp[-3].s, yyvsp[-2].val.i, yyvsp[0].v_d, nested_group_names, yyvsp[-5].val.b); ;
    break;}
case 15:

{ global_params->add(yyvsp[-3].s, yyvsp[-2].val.i, yyvsp[0].v_f, nested_group_names, yyvsp[-5].val.b); ;
    break;}
case 16:

{ global_params->add(yyvsp[-3].s, yyvsp[-2].val.i, yyvsp[0].v_s, nested_group_names, yyvsp[-5].val.b); ;
    break;}
case 17:

{ global_params->add(yyvsp[-6].s, yyvsp[-5].val.i, yyvsp[-4].val.i, yyvsp[-1].vv_i, nested_group_names, yyvsp[-8].val.b); ;
    break;}
case 18:

{ global_params->add(yyvsp[-6].s, yyvsp[-5].val.i, yyvsp[-4].val.i, yyvsp[-1].vv_r, nested_group_names, yyvsp[-8].val.b); ;
    break;}
case 19:

{ global_params->add(yyvsp[-6].s, yyvsp[-5].val.i, yyvsp[-4].val.i, yyvsp[-1].vv_d, nested_group_names, yyvsp[-8].val.b); ;
    break;}
case 20:

{ global_params->add(yyvsp[-6].s, yyvsp[-5].val.i, yyvsp[-4].val.i, yyvsp[-1].vv_f, nested_group_names, yyvsp[-8].val.b); ;
    break;}
case 21:

{ global_params->add(yyvsp[-6].s, yyvsp[-5].val.i, yyvsp[-4].val.i, yyvsp[-1].vv_s, nested_group_names, yyvsp[-8].val.b); ;
    break;}
case 22:

{ yyval.val.b = false;  ;
    break;}
case 23:

{ yyval.val.b = true;   ;
    break;}
case 25:

{ yyval.val.i = global_params->get<int>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 26:

{ yyval.val.i = yyvsp[-2].val.i + yyvsp[0].val.i; ;
    break;}
case 27:

{ yyval.val.i = yyvsp[-2].val.i - yyvsp[0].val.i; ;
    break;}
case 28:

{ yyval.val.i = yyvsp[-2].val.i * yyvsp[0].val.i; ;
    break;}
case 29:

{ if( yyvsp[0].val.i ) yyval.val.i = yyvsp[-2].val.i / yyvsp[0].val.i; else { yyval.val.i=1; yyerror("Error, division by zero"); } ;
    break;}
case 30:

{ if( yyvsp[0].val.i ) yyval.val.i = yyvsp[-2].val.i % yyvsp[0].val.i; else { yyval.val.i=1; yyerror("Error, % by zero"); } ;
    break;}
case 31:

{ yyval.val.i =   yyvsp[0].val.i; ;
    break;}
case 32:

{ yyval.val.i = - yyvsp[0].val.i; ;
    break;}
case 33:

{ int i = yyvsp[0].val.i; yyval.val.i = 1; while( i-- > 0 ) yyval.val.i *= yyvsp[-2].val.i; i = yyvsp[0].val.i; while( i++ < 0 ) { 
    if( yyvsp[-2].val.i ) yyval.val.i /= yyvsp[-2].val.i; else { yyerror("Error, 0 ^ neg"); } } ;
    break;}
case 34:

{ yyval.val.i =   yyvsp[-1].val.i; ;
    break;}
case 35:

{ yyval.val.r = (real)yyvsp[0].val.d; ;
    break;}
case 36:

{ yyval.val.r = global_params->get<real>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 37:

{ yyval.val.r = yyvsp[-2].val.r + yyvsp[0].val.r; ;
    break;}
case 38:

{ yyval.val.r = yyvsp[-2].val.r - yyvsp[0].val.r; ;
    break;}
case 39:

{ yyval.val.r = yyvsp[-2].val.r * yyvsp[0].val.r; ;
    break;}
case 40:

{ if( yyvsp[0].val.r ) yyval.val.r = yyvsp[-2].val.r / yyvsp[0].val.r; else { yyval.val.r=1.0; yyerror("Error, division by zero"); } ;
    break;}
case 41:

{ yyval.val.r =   yyvsp[0].val.r; ;
    break;}
case 42:

{ yyval.val.r = - yyvsp[0].val.r; ;
    break;}
case 43:

{ yyval.val.r =   yyvsp[-1].val.r; ;
    break;}
case 44:

{ yyval.val.d = (double)yyvsp[0].val.d; ;
    break;}
case 45:

{ yyval.val.d = global_params->get<double>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 46:

{ yyval.val.d = yyvsp[-2].val.d + yyvsp[0].val.d; ;
    break;}
case 47:

{ yyval.val.d = yyvsp[-2].val.d - yyvsp[0].val.d; ;
    break;}
case 48:

{ yyval.val.d = yyvsp[-2].val.d * yyvsp[0].val.d; ;
    break;}
case 49:

{ if( yyvsp[0].val.d ) yyval.val.d = yyvsp[-2].val.d / yyvsp[0].val.d; else { yyval.val.d=1.0; yyerror("Error, division by zero"); } ;
    break;}
case 50:

{ yyval.val.d =   yyvsp[0].val.d; ;
    break;}
case 51:

{ yyval.val.d = - yyvsp[0].val.d; ;
    break;}
case 52:

{ yyval.val.d =   yyvsp[-1].val.d; ;
    break;}
case 53:

{ yyval.val.f = (float)yyvsp[0].val.d; ;
    break;}
case 54:

{ yyval.val.f = global_params->get<float>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 55:

{ yyval.val.f = yyvsp[-2].val.f + yyvsp[0].val.f; ;
    break;}
case 56:

{ yyval.val.f = yyvsp[-2].val.f - yyvsp[0].val.f; ;
    break;}
case 57:

{ yyval.val.f = yyvsp[-2].val.f * yyvsp[0].val.f; ;
    break;}
case 58:

{ if( yyvsp[0].val.f ) yyval.val.f = yyvsp[-2].val.f / yyvsp[0].val.f; else { yyval.val.f=1.0; yyerror("Error, division by zero"); } ;
    break;}
case 59:

{ yyval.val.f =   yyvsp[0].val.f; ;
    break;}
case 60:

{ yyval.val.f = - yyvsp[0].val.f; ;
    break;}
case 61:

{ yyval.val.f =   yyvsp[-1].val.f; ;
    break;}
case 63:

{ yyval.s = global_params->get<std::string>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 65:

{ yyval.s = yyvsp[-2].s + global_params->get<std::string>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 66:

{ yyval.s = global_params->get<std::string>(yyvsp[-2].v_s, nested_group_names) + yyvsp[0].s; ;
    break;}
case 67:

{ yyval.s = yyvsp[-2].s + yyvsp[0].s; ;
    break;}
case 68:

{ yyval.s = yyvsp[-2].s + global_params->get<std::string>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 70:

{ yyval.s += yyvsp[0].s; ;
    break;}
case 71:

{ yyval.v_i.resize(0); ;
    break;}
case 72:

{ yyval.v_i = yyvsp[-1].v_i; ;
    break;}
case 73:

{ yyval.v_i.push_back( yyvsp[0].val.i ); ;
    break;}
case 75:

{ yyval.v_i.push_back( yyvsp[0].val.i ); ;
    break;}
case 76:

{ for( size_t i = 0; i != yyvsp[0].v_i.size(); ++i ) yyval.v_i.push_back( yyvsp[0].v_i[i] ); ;
    break;}
case 78:

{ for( int i = yyvsp[-2].val.i; i <= yyvsp[0].val.i; ++i ) yyval.v_i.push_back( i ); ;
    break;}
case 79:

{ yyval.v_r.resize(0); ;
    break;}
case 80:

{ yyval.v_r = yyvsp[-1].v_r; ;
    break;}
case 81:

{ yyval.v_r.push_back( yyvsp[0].val.r ); ;
    break;}
case 82:

{ yyval.v_r.push_back( yyvsp[0].val.r ); ;
    break;}
case 84:

{ yyval.v_d.resize(0); ;
    break;}
case 85:

{ yyval.v_d = yyvsp[-1].v_d; ;
    break;}
case 86:

{ yyval.v_d.push_back( yyvsp[0].val.d ); ;
    break;}
case 87:

{ yyval.v_d.push_back( yyvsp[0].val.d ); ;
    break;}
case 89:

{ yyval.v_f.resize(0); ;
    break;}
case 90:

{ yyval.v_f = yyvsp[-1].v_f; ;
    break;}
case 91:

{ yyval.v_f.push_back( yyvsp[0].val.f ); ;
    break;}
case 92:

{ yyval.v_f.push_back( yyvsp[0].val.f ); ;
    break;}
case 94:

{ yyval.v_s.resize(0); ;
    break;}
case 95:

{ yyval.v_s = yyvsp[-1].v_s; ;
    break;}
case 96:

{ yyval.v_s.push_back( yyvsp[0].s ); ;
    break;}
case 97:

{ yyval.v_s.resize(0); yyval.v_s.push_back( global_params->get<std::string>(yyvsp[0].v_s, nested_group_names) ); ;
    break;}
case 98:

{ yyval.v_s.push_back( yyvsp[0].s ); ;
    break;}
case 99:

{ yyval.v_s.push_back( global_params->get<std::string>(yyvsp[0].v_s, nested_group_names) ); ;
    break;}
case 101:

{ yyval.val.i = 0; ;
    break;}
case 102:

{ yyval.val.i=yyvsp[-1].val.i; if(yyval.val.i <= 0) { std::cout << "\nArray size = " << yyval.val.i << "."; yyerror("error, array size <= 0"); } ;
    break;}
case 104:

{ bool iscons = global_params->is_const<int>(yyvsp[0].v_s, nested_group_names);
    if( ! iscons ) { std::cout << "\nVar: \"" << "::...\"" << yyvsp[0].v_s.back() << "\"."; yyerror("error, need constant expr"); }
    yyval.val.i = global_params->get<int>(yyvsp[0].v_s, nested_group_names); ;
    break;}
case 105:

{ yyval.val.i = yyvsp[-2].val.i + yyvsp[0].val.i; ;
    break;}
case 106:

{ yyval.val.i = yyvsp[-2].val.i - yyvsp[0].val.i; ;
    break;}
case 107:

{ yyval.val.i = yyvsp[-2].val.i * yyvsp[0].val.i; ;
    break;}
case 108:

{ if( yyvsp[0].val.i ) yyval.val.i = yyvsp[-2].val.i / yyvsp[0].val.i; else{ yyval.val.i=1; yyerror("error, division by zero"); } ;
    break;}
case 109:

{ if( yyvsp[0].val.i ) yyval.val.i = yyvsp[-2].val.i % yyvsp[0].val.i; else{ yyval.val.i=1; yyerror("error, % by zero"); } ;
    break;}
case 110:

{ yyval.val.i =   yyvsp[0].val.i;   ;
    break;}
case 111:

{ yyval.val.i = - yyvsp[0].val.i;   ;
    break;}
case 112:

{ int i = yyvsp[0].val.i; yyval.val.i = 1; while( i-- > 0 ) yyval.val.i *= yyvsp[-2].val.i; i = yyvsp[0].val.i; while( i++ < 0 ) { 
    if( yyvsp[-2].val.i ) yyval.val.i /= yyvsp[-2].val.i; else{ yyerror("error, 0 ^ neg"); } } ;
    break;}
case 113:

{ yyval.val.i = yyvsp[-1].val.i; ;
    break;}
case 114:

{ yyval.v_s.push_back( yyvsp[0].s ); ;
    break;}
case 115:

{ yyval.v_s.push_back( yyvsp[0].s ); ;
    break;}
case 116:

{ global_params->insert_group(yyvsp[-1].s, nested_group_names); nested_group_names.push_back(yyvsp[-1].s); ;
    break;}
case 117:

{ nested_group_names.pop_back(); ;
    break;}
case 118:

{ if( nested_group_names.empty() ) { yyerror("error, top group must have a name!"); }
    std::string tmp = global_params->insert_group(nested_group_names); nested_group_names.push_back(tmp); ;
    break;}
case 119:

{ nested_group_names.pop_back(); ;
    break;}
case 120:

{ yyval.vv_i.push_back( yyvsp[0].v_i ); ;
    break;}
case 121:

{ yyval.vv_i.push_back( yyvsp[0].v_i ); ;
    break;}
case 123:

{ yyval.vv_r.push_back( yyvsp[0].v_r ); ;
    break;}
case 124:

{ yyval.vv_r.push_back( yyvsp[0].v_r ); ;
    break;}
case 126:

{  yyval.vv_d.push_back( yyvsp[0].v_d ); ;
    break;}
case 127:

{ yyval.vv_d.push_back( yyvsp[0].v_d ); ;
    break;}
case 129:

{ yyval.vv_f.push_back( yyvsp[0].v_f ); ;
    break;}
case 130:

{ yyval.vv_f.push_back( yyvsp[0].v_f ); ;
    break;}
case 132:

{ yyval.vv_s.push_back( yyvsp[0].v_s ); ;
    break;}
case 133:

{ yyval.vv_s.push_back( yyvsp[0].v_s ); ;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */


  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YYLSP_NEEDED
  yylsp -= yylen;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
    fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YYLSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = yylloc.first_line;
      yylsp->first_column = yylloc.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  goto yynewstate;

yyerrlab:   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++yynerrs;

#ifdef YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
    {
      int size = 0;
      char *msg;
      int x, count;

      count = 0;
      /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
      for (x = (yyn < 0 ? -yyn : 0);
           x < (sizeof(yytname) / sizeof(char *)); x++)
        if (yycheck[x + yyn] == x)
          size += strlen(yytname[x]) + 15, count++;
      msg = (char *) malloc(size + 15);
      if (msg != 0)
        {
          strcpy(msg, "parse error");

          if (count < 5)
        {
          count = 0;
          for (x = (yyn < 0 ? -yyn : 0);
               x < (sizeof(yytname) / sizeof(char *)); x++)
            if (yycheck[x + yyn] == x)
              {
            strcat(msg, count == 0 ? ", expecting `" : " or `");
            strcat(msg, yytname[x]);
            strcat(msg, "'");
            count++;
              }
        }
          yyerror(msg);
          free(msg);
        }
      else
        yyerror ("parse error; also virtual memory exceeded");
    }
      else
#endif /* YYERROR_VERBOSE */
    yyerror("parse error");
    }

  goto yyerrlab1;
yyerrlab1:   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (yychar == YYEOF)
    YYABORT;

#if YYDEBUG != 0
      if (yydebug)
    fprintf(stderr, "Discarding token %d (%s).\n", yychar, yytname[yychar1]);
#endif

      yychar = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;      /* Each real token shifted decrements this */

  goto yyerrhandle;

yyerrdefault:  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) goto yydefault;
#endif

yyerrpop:   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YYLSP_NEEDED
  yylsp--;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
    fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

yyerrhandle:

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yyerrdefault;

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    goto yyerrdefault;

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
    goto yyerrpop;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrpop;

  if (yyn == YYFINAL)
    YYACCEPT;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  yystate = yyn;
  goto yynewstate;
}



int read_parameters( const std::string & filename, parameter_space::parameters & params )
{
#if YYDEBUG != 0
    extern int yydebug;
    yydebug = 1;
#endif

    global_params = & params;
    if( new_file(filename) ) yyparse();
    cur_buf_stack = _nullptr;
    global_params = NULL;
    return 0;
}

void yyerror( const std::string & msg )
{  
    std::cout << "\nEncountered: " << msg << " at line " << line_no << " of file \"" << cur_filename << "\"." << std::endl;

    abort();
}
