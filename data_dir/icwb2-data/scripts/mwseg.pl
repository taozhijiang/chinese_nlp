#!/usr/bin/perl -w

###########################################################################
#                                                                         #
#                               SIGHAN                                    #
#                         Copyright (c) 2003                              #
#                        All Rights Reserved.                             #
#                                                                         #
#  Permission is hereby granted, free of charge, to use and distribute    #
#  this software and its documentation without restriction, including     #
#  without limitation the rights to use, copy, modify, merge, publish,    #
#  distribute, sublicense, and/or sell copies of this work, and to        #
#  permit persons to whom this work is furnished to do so, subject to     #
#  the following conditions:                                              #
#   1. The code must retain the above copyright notice, this list of      #
#      conditions and the following disclaimer.                           #
#   2. Any modifications must be clearly marked as such.                  #
#   3. Original authors' names are not deleted.                           #
#   4. The authors' names are not used to endorse or promote products     #
#      derived from this software without specific prior written          #
#      permission.                                                        #
#                                                                         #
#  SIGHAN AND THE CONTRIBUTORS TO THIS WORK DISCLAIM ALL WARRANTIES       #
#  WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF      #
#  MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL SIGHAN NOR THE          #
#  CONTRIBUTORS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL      #
#  DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA     #
#  OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER      #
#  TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR       #
#  PERFORMANCE OF THIS SOFTWARE.                                          #
#                                                                         #
###########################################################################
#                                                                         #
# Author: Richard Sproat (rws@research.att.com)                           #
#                                                                         #
###########################################################################

$USAGE = "Usage:\t$0 dictionary\n\t";

if (@ARGV < 1) {print "$USAGE\n"; exit;}

%dict = ();
$maxwlen = 0;

open (S, $ARGV[0]) or die "$ARGV[0]: $!\n";
while (<S>) { 
    chop; 
    $dict{$_} = 1; 
    my $l = length($_);
    $maxwlen = $l if $l > $maxwlen;
}
close (S);

shift @ARGV;

$n = 0;
while (<>) {
    chop;
    s/\s*//g;
    my $text = $_;
    while ($text ne "") {
	$sub = substr($text, 0, $maxwlen);
	while ($sub ne "") {
	    if ($dict{$sub}) {
		print "$sub ";
		for (my $i = 0; $i < length($sub); ++$i) {
		    $text =~ s/^.//;
		}
		last;
	    }
	    $sub =~ s/.$//;
	}
	if ($sub eq "")  {
	    if ($text =~ /^([\x21-\x7e])/) {
		print "$1 ";
		$text =~ s/^.//;
	    }
	    elsif ($text =~ /^([^\x21-\x7e].)/) {
		print "$1 ";
		$text =~ s/^..//;
	    }
	    else { ## shouldn't happen
		print STDERR "Oops: shouldn't be here: $n\n";
		print "$1 ";
		$text =~ s/^.//;
	    }
	}
    }
    print "\n";
    ++$n;
}

exit(0);
