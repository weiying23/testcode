/*  Copyright 2014 - UVSQ
    Authors list: LoÃ¯c ThÃ©bault, Eric Petit

    This file is part of the DC-lib.

    DC-lib is free software: you can redistribute it and/or modify it under the
    terms of the GNU Lesser General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later version.

    DC-lib is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along with
    the DC-lib. If not, see <http://www.gnu.org/licenses/>. */

#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

#ifdef DC_VEC
// Create coloring permutation array with full vectorial colors stored first &
// return the index of the last element in a full vectorial color
int create_coloring_permutation (int *perm, int *part, int *card, int size,
                                 int nbColors);
#endif
#endif